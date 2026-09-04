"""
Phase 2F haircut similarity engine with a haircut-family gate plus fine-grained fade analyzer.

Goals:
- preserve Phase 2C robustness to hair colour, skin tone, background and profile direction;
- explicitly separate FULL FADE vs TAPER FADE vs CLASSIC TAPER;
- explicitly score fade height, side coverage and shortest/base length;
- compare a side-of-head crop so the top/back style cannot drown out fade differences;
- expose diagnostics so you can see WHY two cuts got different scores;
- avoid visual-score saturation and make fade-family disagreements matter even when zero-shot confidence is modest;
- prevent high generic visual similarity from overriding an incompatible haircut family (for example mullet/wolf vs crop/buzz).

This remains an MVP heuristic built on zero-shot OpenCLIP. It is not a fairness
or quality guarantee and must be calibrated on a diverse labelled haircut set
before any real-money deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Tuple

import open_clip
import torch
from PIL import Image, ImageFilter, ImageOps

from config import (
    CLIP_MODEL,
    CLIP_PRETRAINED,
    SIMILARITY_CEILING,
    SIMILARITY_FLOOR,
)


# General haircut attributes. Race, skin tone, gender and identity labels are
# intentionally excluded from the settlement scorer.
ATTRIBUTE_GROUPS: Dict[str, List[Tuple[str, str]]] = {
    "style": [
        ("mullet", "a mullet haircut, shorter through the sides with visibly longer hair at the back"),
        ("wolf_cut", "a wolf cut haircut with layered volume and extra length at the back"),
        ("textured_crop", "a short textured crop haircut"),
        ("crew_cut", "a crew cut haircut"),
        ("buzz_cut", "a very short buzz cut haircut"),
        ("slick_back", "a slicked back haircut"),
        ("pompadour", "a pompadour haircut with height at the front"),
        ("long_hair", "a long hairstyle extending clearly below the ears"),
    ],
    "top_texture": [
        ("textured", "a choppy textured top"),
        ("straight", "a mostly straight smooth top"),
        ("wavy", "a wavy top"),
        ("curly", "a curly top"),
        ("slicked", "the top slicked back"),
    ],
    "back_length": [
        ("very_short", "a very short back and nape"),
        ("short", "a short back"),
        ("medium", "medium length at the back"),
        ("long", "clearly long hair at the back or nape"),
    ],
}

ATTRIBUTE_GROUP_WEIGHTS = {
    "style": 0.45,
    "top_texture": 0.30,
    "back_length": 0.25,
}

# Phase 2F: discrete haircut-family compatibility. Distribution overlap alone is
# too forgiving: CLIP can consider a buzz/crop and a mullet visually similar
# because both images contain the same head/pose/fade. This matrix makes the
# predicted haircut family materially affect settlement.
STYLE_MATCH = {
    ("mullet", "mullet"): 100.0,
    ("wolf_cut", "wolf_cut"): 100.0,
    ("textured_crop", "textured_crop"): 100.0,
    ("crew_cut", "crew_cut"): 100.0,
    ("buzz_cut", "buzz_cut"): 100.0,
    ("slick_back", "slick_back"): 100.0,
    ("pompadour", "pompadour"): 100.0,
    ("long_hair", "long_hair"): 100.0,

    ("mullet", "wolf_cut"): 82.0,
    ("wolf_cut", "mullet"): 82.0,
    ("textured_crop", "crew_cut"): 70.0,
    ("crew_cut", "textured_crop"): 70.0,
    ("crew_cut", "buzz_cut"): 68.0,
    ("buzz_cut", "crew_cut"): 68.0,
    ("textured_crop", "buzz_cut"): 45.0,
    ("buzz_cut", "textured_crop"): 45.0,
    ("slick_back", "pompadour"): 70.0,
    ("pompadour", "slick_back"): 70.0,
    ("wolf_cut", "long_hair"): 58.0,
    ("long_hair", "wolf_cut"): 58.0,
    ("mullet", "long_hair"): 50.0,
    ("long_hair", "mullet"): 50.0,

    # Deliberately incompatible families.
    ("mullet", "textured_crop"): 24.0,
    ("textured_crop", "mullet"): 24.0,
    ("wolf_cut", "textured_crop"): 30.0,
    ("textured_crop", "wolf_cut"): 30.0,
    ("mullet", "crew_cut"): 12.0,
    ("crew_cut", "mullet"): 12.0,
    ("wolf_cut", "crew_cut"): 16.0,
    ("crew_cut", "wolf_cut"): 16.0,
    ("mullet", "buzz_cut"): 4.0,
    ("buzz_cut", "mullet"): 4.0,
    ("wolf_cut", "buzz_cut"): 6.0,
    ("buzz_cut", "wolf_cut"): 6.0,
    ("long_hair", "buzz_cut"): 2.0,
    ("buzz_cut", "long_hair"): 2.0,
    ("long_hair", "crew_cut"): 8.0,
    ("crew_cut", "long_hair"): 8.0,
    ("long_hair", "textured_crop"): 15.0,
    ("textured_crop", "long_hair"): 15.0,
}

BACK_LENGTH_MATCH = {
    ("very_short", "very_short"): 100.0,
    ("short", "short"): 100.0,
    ("medium", "medium"): 100.0,
    ("long", "long"): 100.0,
    ("very_short", "short"): 72.0,
    ("short", "very_short"): 72.0,
    ("short", "medium"): 55.0,
    ("medium", "short"): 55.0,
    ("medium", "long"): 62.0,
    ("long", "medium"): 62.0,
    ("very_short", "medium"): 25.0,
    ("medium", "very_short"): 25.0,
    ("short", "long"): 20.0,
    ("long", "short"): 20.0,
    ("very_short", "long"): 4.0,
    ("long", "very_short"): 4.0,
}


# Fine-grained fade concepts. These are deliberately split so "low fade" and
# "low taper fade" are NOT treated as one generic fade label.
FADE_GROUPS: Dict[str, List[Tuple[str, str]]] = {
    "fade_family": [
        (
            "full_fade",
            "a full side fade where the short-to-long blend continues across most of the side and around the ear",
        ),
        (
            "taper_fade",
            "a taper fade concentrated near the temple and lower edge while noticeably more length remains through the side",
        ),
        (
            "classic_taper",
            "a classic taper with gradual shortening mainly at the temple and neckline, not a full side fade",
        ),
        ("no_fade", "no visible fade or taper on the side"),
    ],
    "fade_height": [
        ("low", "a low fade or taper that begins close to the ear and lower temple"),
        ("mid", "a mid fade that begins around the middle of the side of the head"),
        ("high", "a high fade that begins high on the side of the head"),
        ("none", "no visible fade height because there is no fade or taper"),
    ],
    "fade_coverage": [
        (
            "temple_only",
            "the blend is tightly concentrated at the temple with most side hair left intact",
        ),
        (
            "temple_and_ear",
            "the blend covers the temple and the area immediately around the ear but not the whole side",
        ),
        (
            "full_side",
            "the blend extends across most of the side of the head and wraps broadly around the ear",
        ),
        ("none", "there is no visible faded or tapered side coverage"),
    ],
    "base_length": [
        ("skin", "the shortest part is shaved to skin or nearly skin"),
        ("very_short", "the shortest part is clipper-short with very little visible hair"),
        ("shadow", "the shortest part leaves a soft dark shadow of hair"),
        ("long_taper", "the shortest area still keeps noticeable hair length as in a gentle taper"),
        ("none", "there is no distinct shortest fade or taper section"),
    ],
    "blend_quality": [
        ("clean", "a technically clean fade with a smooth seamless blend and no visible weight line"),
        ("slight_banding", "a mostly clean fade with a faint visible transition band or weight line"),
        ("harsh_banding", "a fade with a clearly harsh transition line or obvious dark band"),
        ("patchy", "a patchy uneven fade with inconsistent density or poorly blended areas"),
    ],
}

FADE_GROUP_WEIGHTS = {
    "fade_family": 0.40,
    "fade_height": 0.20,
    "fade_coverage": 0.20,
    "base_length": 0.10,
    "blend_quality": 0.10,
}

FADE_FAMILY_MATCH = {
    ("full_fade", "full_fade"): 100.0,
    ("taper_fade", "taper_fade"): 100.0,
    ("classic_taper", "classic_taper"): 100.0,
    ("no_fade", "no_fade"): 100.0,
    ("full_fade", "taper_fade"): 48.0,
    ("taper_fade", "full_fade"): 48.0,
    ("taper_fade", "classic_taper"): 58.0,
    ("classic_taper", "taper_fade"): 58.0,
    ("full_fade", "classic_taper"): 28.0,
    ("classic_taper", "full_fade"): 28.0,
    ("no_fade", "classic_taper"): 42.0,
    ("classic_taper", "no_fade"): 42.0,
    ("no_fade", "taper_fade"): 20.0,
    ("taper_fade", "no_fade"): 20.0,
    ("no_fade", "full_fade"): 8.0,
    ("full_fade", "no_fade"): 8.0,
}

FADE_HEIGHT_MATCH = {
    ("low", "low"): 100.0,
    ("mid", "mid"): 100.0,
    ("high", "high"): 100.0,
    ("none", "none"): 100.0,
    ("low", "mid"): 58.0,
    ("mid", "low"): 58.0,
    ("mid", "high"): 55.0,
    ("high", "mid"): 55.0,
    ("low", "high"): 18.0,
    ("high", "low"): 18.0,
}

FADE_COVERAGE_MATCH = {
    ("temple_only", "temple_only"): 100.0,
    ("temple_and_ear", "temple_and_ear"): 100.0,
    ("full_side", "full_side"): 100.0,
    ("none", "none"): 100.0,
    ("temple_only", "temple_and_ear"): 66.0,
    ("temple_and_ear", "temple_only"): 66.0,
    ("temple_and_ear", "full_side"): 55.0,
    ("full_side", "temple_and_ear"): 55.0,
    ("temple_only", "full_side"): 25.0,
    ("full_side", "temple_only"): 25.0,
}

VISUAL_SCORE_ANCHORS = [
    (0.45, 0.0),
    (0.58, 30.0),
    (0.66, 60.0),
    (0.74, 82.0),
    (0.84, 93.0),
    (0.95, 100.0),
]

FADE_VISUAL_SCORE_ANCHORS = [
    (0.50, 0.0),
    (0.62, 30.0),
    (0.72, 55.0),
    (0.82, 75.0),
    (0.90, 90.0),
    (0.97, 100.0),
]

BLEND_QUALITY_VALUE = {
    "clean": 100.0,
    "slight_banding": 72.0,
    "harsh_banding": 38.0,
    "patchy": 20.0,
}

FADE_VISIBILITY_CHOICES: List[Tuple[str, str]] = [
    (
        "visible",
        "a close-up barber photo where the side fade or taper around the temple and ear is clearly visible",
    ),
    (
        "hidden",
        "a haircut photo where the side fade or taper is hidden, cropped out, or not clearly visible",
    ),
]


@dataclass
class ScoreResult:
    score: int
    raw_similarity: float
    view_similarities: Dict[str, float]
    component_scores: Dict[str, float]
    attribute_similarities: Dict[str, float]
    attribute_predictions: Dict[str, Dict[str, str]]
    style_gate: Dict[str, object]
    fade_analysis: Dict[str, object]
    device: str
    model: str


class HaircutScorer:
    """OpenCLIP haircut scorer with a dedicated side/fade analyzer."""

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(
            f"[AI] Loading Phase 2F {CLIP_MODEL} / {CLIP_PRETRAINED} "
            f"on {self.device}..."
        )

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL,
            pretrained=CLIP_PRETRAINED,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
        self.attribute_text_features = self._build_text_features(ATTRIBUTE_GROUPS)
        self.fade_text_features = self._build_text_features(FADE_GROUPS)
        self.fade_visibility_features = self._encode_prompt_choices(
            FADE_VISIBILITY_CHOICES
        )

        print("[AI] Phase 2F haircut-family gate + fade analyzer ready.")

    @staticmethod
    def _load_image(image_bytes: bytes) -> Image.Image:
        image = Image.open(BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")

    @staticmethod
    def _hair_focus_crop(image: Image.Image) -> Image.Image:
        """Upper/head region preserving side and back length."""
        width, height = image.size
        left = int(width * 0.04)
        right = max(left + 1, int(width * 0.96))
        bottom = max(1, int(height * 0.76))
        return image.crop((left, 0, right, bottom))

    @staticmethod
    def _fade_side_candidates(image: Image.Image) -> List[Image.Image]:
        """
        Produce overlapping left/right side candidates.

        We do not know whether the photo is a left or right profile, so the
        model chooses whichever crop makes a fade/taper most visible.
        """
        width, height = image.size
        top = int(height * 0.10)
        bottom = max(top + 1, int(height * 0.80))

        left = image.crop((0, top, max(1, int(width * 0.66)), bottom))
        right = image.crop((int(width * 0.34), top, width, bottom))
        return [left, right]

    @staticmethod
    def _to_gray_rgb(image: Image.Image) -> Image.Image:
        return ImageOps.autocontrast(image.convert("L")).convert("RGB")

    @staticmethod
    def _to_shape_view(image: Image.Image) -> Image.Image:
        gray = ImageOps.autocontrast(image.convert("L"))
        radius = max(1.5, min(image.size) / 90.0)
        blurred = gray.filter(ImageFilter.GaussianBlur(radius=radius))
        return ImageOps.autocontrast(blurred).convert("RGB")

    @staticmethod
    def _flip(image: Image.Image) -> Image.Image:
        return ImageOps.mirror(image)

    def _make_views(self, image: Image.Image) -> Dict[str, Image.Image]:
        hair = self._hair_focus_crop(image)
        return {
            "hair_gray": self._to_gray_rgb(hair),
            "hair_shape": self._to_shape_view(hair),
            "full_gray": self._to_gray_rgb(image),
        }

    def _embed(self, images: List[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self.preprocess(image) for image in images]).to(
            self.device
        )
        with torch.inference_mode():
            features = self.model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    def _encode_prompt_choices(
        self,
        choices: List[Tuple[str, str]],
    ) -> torch.Tensor:
        prompts = [
            f"A close-up professional barber photograph clearly showing {description}."
            for _, description in choices
        ]
        with torch.inference_mode():
            tokens = self.tokenizer(prompts).to(self.device)
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    def _build_text_features(
        self,
        groups: Dict[str, List[Tuple[str, str]]],
    ) -> Dict[str, torch.Tensor]:
        return {
            group: self._encode_prompt_choices(choices)
            for group, choices in groups.items()
        }

    @staticmethod
    def _interpolate_score(raw_similarity: float, anchors: List[Tuple[float, float]]) -> float:
        """Piecewise-linear calibration that avoids early saturation."""
        if raw_similarity <= anchors[0][0]:
            return anchors[0][1]
        if raw_similarity >= anchors[-1][0]:
            return anchors[-1][1]
        for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
            if x0 <= raw_similarity <= x1:
                t = (raw_similarity - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        return anchors[-1][1]

    @staticmethod
    def _calibrate_similarity(raw_similarity: float) -> float:
        return HaircutScorer._interpolate_score(raw_similarity, VISUAL_SCORE_ANCHORS)

    @staticmethod
    def _calibrate_fade_visual(raw_similarity: float) -> float:
        return HaircutScorer._interpolate_score(raw_similarity, FADE_VISUAL_SCORE_ANCHORS)

    @staticmethod
    def _pair_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a * b).sum().detach().cpu().item())

    def _orientation_invariant_similarity(
        self,
        reference_view: Image.Image,
        result_view: Image.Image,
    ) -> float:
        embeddings = self._embed(
            [reference_view, result_view, self._flip(result_view)]
        )
        reference_feature = embeddings[0]
        direct = self._pair_similarity(reference_feature, embeddings[1])
        mirrored = self._pair_similarity(reference_feature, embeddings[2])
        return max(direct, mirrored)

    def _orientation_averaged_feature(self, image: Image.Image) -> torch.Tensor:
        embeddings = self._embed([image, self._flip(image)])
        feature = embeddings.mean(dim=0, keepdim=True)
        feature = feature / feature.norm(dim=-1, keepdim=True)
        return feature[0]

    def _profile_from_feature(
        self,
        feature: torch.Tensor,
        groups: Dict[str, List[Tuple[str, str]]],
        text_features: Dict[str, torch.Tensor],
        temperature: float,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, str], Dict[str, float]]:
        distributions: Dict[str, torch.Tensor] = {}
        predictions: Dict[str, str] = {}
        confidences: Dict[str, float] = {}

        for group, choices in groups.items():
            logits = temperature * (feature @ text_features[group].T)
            probs = torch.softmax(logits, dim=-1)
            distributions[group] = probs
            top_index = int(torch.argmax(probs).detach().cpu().item())
            predictions[group] = choices[top_index][0]
            confidences[group] = float(probs[top_index].detach().cpu().item())

        return distributions, predictions, confidences

    def _attribute_profile(
        self,
        image: Image.Image,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, str], Dict[str, float]]:
        feature = self._orientation_averaged_feature(image)
        return self._profile_from_feature(
            feature,
            ATTRIBUTE_GROUPS,
            self.attribute_text_features,
            temperature=35.0,
        )

    def _select_fade_view(
        self,
        image: Image.Image,
    ) -> Tuple[Image.Image, float]:
        candidates = [
            self._to_gray_rgb(candidate)
            for candidate in self._fade_side_candidates(image)
        ]
        features = [self._orientation_averaged_feature(c) for c in candidates]

        best_index = 0
        best_visible_probability = -1.0

        for index, feature in enumerate(features):
            logits = 30.0 * (feature @ self.fade_visibility_features.T)
            probs = torch.softmax(logits, dim=-1)
            visible_probability = float(probs[0].detach().cpu().item())
            if visible_probability > best_visible_probability:
                best_visible_probability = visible_probability
                best_index = index

        return candidates[best_index], best_visible_probability

    def _fade_profile(
        self,
        image: Image.Image,
    ) -> Tuple[
        Image.Image,
        Dict[str, torch.Tensor],
        Dict[str, str],
        Dict[str, float],
        float,
    ]:
        fade_view, visibility = self._select_fade_view(image)
        feature = self._orientation_averaged_feature(fade_view)
        distributions, predictions, confidences = self._profile_from_feature(
            feature,
            FADE_GROUPS,
            self.fade_text_features,
            temperature=42.0,
        )
        return fade_view, distributions, predictions, confidences, visibility

    @staticmethod
    def _distribution_overlap(a: torch.Tensor, b: torch.Tensor) -> float:
        overlap = torch.minimum(a, b).sum()
        return float(overlap.detach().cpu().item())

    @staticmethod
    def _expected_blend_quality(
        distribution: torch.Tensor,
    ) -> float:
        labels = [key for key, _ in FADE_GROUPS["blend_quality"]]
        values = torch.tensor(
            [BLEND_QUALITY_VALUE[label] for label in labels],
            dtype=distribution.dtype,
            device=distribution.device,
        )
        return float((distribution * values).sum().detach().cpu().item())

    @staticmethod
    def _label_match_score(
        reference_label: str,
        result_label: str,
        matrix: Dict[Tuple[str, str], float],
        same_default: float = 100.0,
        different_default: float = 45.0,
    ) -> float:
        if reference_label == result_label:
            return same_default
        return matrix.get((reference_label, result_label), different_default)

    @staticmethod
    def _confidence_margin(distribution: torch.Tensor) -> float:
        values, _ = torch.sort(distribution, descending=True)
        if values.numel() < 2:
            return 1.0
        return float((values[0] - values[1]).detach().cpu().item())

    @staticmethod
    def _fade_mismatch_penalty(
        reference_predictions: Dict[str, str],
        result_predictions: Dict[str, str],
        reference_confidence: Dict[str, float],
        result_confidence: Dict[str, float],
        reference_visibility: float,
        result_visibility: float,
    ) -> Tuple[float, List[str]]:
        penalty = 0.0
        reasons: List[str] = []

        visibility = min(reference_visibility, result_visibility)
        if visibility < 0.45:
            return 0.0, ["fade side not visible enough for strict mismatch penalties"]

        vis_scale = max(0.65, min(1.0, (visibility - 0.45) / 0.30 + 0.65))

        ref_family = reference_predictions["fade_family"]
        res_family = result_predictions["fade_family"]
        if ref_family != res_family:
            pair = {ref_family, res_family}
            if "no_fade" in pair and ("full_fade" in pair or "taper_fade" in pair):
                amount = 16.0
            elif pair == {"full_fade", "taper_fade"}:
                amount = 10.0
            elif pair == {"taper_fade", "classic_taper"}:
                amount = 7.0
            else:
                amount = 9.0
            amount *= vis_scale
            penalty += amount
            reasons.append(f"fade family mismatch: {ref_family} vs {res_family} (-{amount:.1f})")

        ref_height = reference_predictions["fade_height"]
        res_height = result_predictions["fade_height"]
        if ref_height != res_height:
            if "none" in {ref_height, res_height}:
                amount = 6.0
            elif {ref_height, res_height} == {"low", "high"}:
                amount = 8.0
            else:
                amount = 4.5
            amount *= vis_scale
            penalty += amount
            reasons.append(f"fade height mismatch: {ref_height} vs {res_height} (-{amount:.1f})")

        ref_coverage = reference_predictions["fade_coverage"]
        res_coverage = result_predictions["fade_coverage"]
        if ref_coverage != res_coverage:
            if "none" in {ref_coverage, res_coverage}:
                amount = 9.0
            elif {ref_coverage, res_coverage} == {"temple_only", "full_side"}:
                amount = 7.0
            else:
                amount = 4.5
            amount *= vis_scale
            penalty += amount
            reasons.append(f"fade coverage mismatch: {ref_coverage} vs {res_coverage} (-{amount:.1f})")

        ref_base = reference_predictions["base_length"]
        res_base = result_predictions["base_length"]
        if ref_base != res_base:
            amount = 2.5 * vis_scale
            penalty += amount
            reasons.append(f"fade base-length mismatch: {ref_base} vs {res_base} (-{amount:.1f})")

        quality_order = {"patchy": 0, "harsh_banding": 1, "slight_banding": 2, "clean": 3}
        ref_quality = reference_predictions["blend_quality"]
        res_quality = result_predictions["blend_quality"]
        drop = quality_order[ref_quality] - quality_order[res_quality]
        if drop > 0:
            amount = min(7.0, 2.5 * drop) * vis_scale
            penalty += amount
            reasons.append(f"fade blend quality dropped: {ref_quality} vs {res_quality} (-{amount:.1f})")

        return penalty, reasons

    def compare(self, reference_bytes: bytes, result_bytes: bytes) -> ScoreResult:
        reference = self._load_image(reference_bytes)
        result = self._load_image(result_bytes)

        reference_views = self._make_views(reference)
        result_views = self._make_views(result)

        view_similarities: Dict[str, float] = {}
        for name in ("hair_gray", "hair_shape", "full_gray"):
            view_similarities[name] = self._orientation_invariant_similarity(
                reference_views[name], result_views[name]
            )

        raw_similarity = (
            view_similarities["hair_gray"] * 0.62
            + view_similarities["hair_shape"] * 0.30
            + view_similarities["full_gray"] * 0.08
        )
        visual_score = self._calibrate_similarity(raw_similarity)

        # General haircut profile (style / texture / back length).
        ref_profile, ref_predictions, _ = self._attribute_profile(
            reference_views["hair_gray"]
        )
        res_profile, res_predictions, _ = self._attribute_profile(
            result_views["hair_gray"]
        )

        attribute_similarities: Dict[str, float] = {}
        for group in ATTRIBUTE_GROUPS:
            attribute_similarities[group] = self._distribution_overlap(
                ref_profile[group], res_profile[group]
            )

        attribute_predictions = {
            group: {
                "reference": ref_predictions[group],
                "result": res_predictions[group],
            }
            for group in ATTRIBUTE_GROUPS
        }

        # Phase 2F: use discrete style/back-length compatibility as well as
        # soft distribution overlap. This is the core fix for cases where a
        # buzz/crop and a mullet looked generically similar enough to pass.
        style_match = self._label_match_score(
            ref_predictions["style"],
            res_predictions["style"],
            STYLE_MATCH,
            different_default=38.0,
        )
        back_length_match = self._label_match_score(
            ref_predictions["back_length"],
            res_predictions["back_length"],
            BACK_LENGTH_MATCH,
            different_default=45.0,
        )

        style_overlap_score = attribute_similarities["style"] * 100.0
        texture_overlap_score = attribute_similarities["top_texture"] * 100.0
        back_overlap_score = attribute_similarities["back_length"] * 100.0

        style_component = 0.40 * style_overlap_score + 0.60 * style_match
        back_component = 0.50 * back_overlap_score + 0.50 * back_length_match
        attribute_score = (
            0.55 * style_component
            + 0.20 * texture_overlap_score
            + 0.25 * back_component
        )

        # Dedicated side/fade analysis.
        (
            ref_fade_view,
            ref_fade_profile,
            ref_fade_predictions,
            ref_fade_confidence,
            ref_visibility,
        ) = self._fade_profile(reference)
        (
            res_fade_view,
            res_fade_profile,
            res_fade_predictions,
            res_fade_confidence,
            res_visibility,
        ) = self._fade_profile(result)

        fade_visual_similarity = self._orientation_invariant_similarity(
            ref_fade_view, res_fade_view
        )
        fade_visual_score = self._calibrate_fade_visual(fade_visual_similarity)

        fade_detail_similarities: Dict[str, float] = {}
        overlap_semantic_score = 0.0
        for group in FADE_GROUPS:
            similarity = self._distribution_overlap(ref_fade_profile[group], res_fade_profile[group])
            fade_detail_similarities[group] = similarity
            overlap_semantic_score += similarity * 100.0 * FADE_GROUP_WEIGHTS[group]

        family_match = self._label_match_score(
            ref_fade_predictions["fade_family"], res_fade_predictions["fade_family"], FADE_FAMILY_MATCH
        )
        height_match = self._label_match_score(
            ref_fade_predictions["fade_height"], res_fade_predictions["fade_height"], FADE_HEIGHT_MATCH
        )
        coverage_match = self._label_match_score(
            ref_fade_predictions["fade_coverage"], res_fade_predictions["fade_coverage"], FADE_COVERAGE_MATCH
        )
        base_match = 100.0 if ref_fade_predictions["base_length"] == res_fade_predictions["base_length"] else 55.0
        quality_match = 100.0 if ref_fade_predictions["blend_quality"] == res_fade_predictions["blend_quality"] else 62.0

        discrete_semantic_score = (
            0.40 * family_match
            + 0.20 * height_match
            + 0.20 * coverage_match
            + 0.10 * base_match
            + 0.10 * quality_match
        )
        fade_semantic_score = 0.30 * overlap_semantic_score + 0.70 * discrete_semantic_score

        reference_blend_quality = self._expected_blend_quality(ref_fade_profile["blend_quality"])
        result_blend_quality = self._expected_blend_quality(res_fade_profile["blend_quality"])

        mismatch_penalty, mismatch_reasons = self._fade_mismatch_penalty(
            ref_fade_predictions,
            res_fade_predictions,
            ref_fade_confidence,
            res_fade_confidence,
            ref_visibility,
            res_visibility,
        )

        fade_detail_score = (
            0.30 * fade_visual_score
            + 0.70 * fade_semantic_score
            - 0.35 * mismatch_penalty
        )
        fade_detail_score = max(0.0, min(100.0, fade_detail_score))

        # Phase 2F haircut-family gate. A high image-embedding similarity must
        # never overpower a clearly incompatible haircut family. Fade weight is
        # also reduced when the side/fade is not actually visible in both photos.
        style_gate_penalty = 0.0
        style_gate_reasons: List[str] = []
        score_cap = 100.0

        if style_match <= 30.0 and back_length_match <= 55.0:
            style_gate_penalty = 18.0
            score_cap = 68.0
            style_gate_reasons.append(
                f"severe haircut-family + back-length mismatch: "
                f"{ref_predictions['style']} vs {res_predictions['style']}; "
                f"{ref_predictions['back_length']} vs {res_predictions['back_length']}"
            )
        elif style_match <= 30.0:
            style_gate_penalty = 11.0
            score_cap = 74.0
            style_gate_reasons.append(
                f"major haircut-family mismatch: {ref_predictions['style']} vs {res_predictions['style']}"
            )
        elif style_match <= 45.0 and back_length_match <= 55.0:
            style_gate_penalty = 7.0
            score_cap = 78.0
            style_gate_reasons.append(
                f"meaningful style/length mismatch: {ref_predictions['style']} vs {res_predictions['style']}"
            )

        fade_visibility = min(ref_visibility, res_visibility)
        if fade_visibility < 0.45:
            effective_weights = {"visual": 0.20, "attributes": 0.55, "fade_detail": 0.25}
        else:
            effective_weights = {"visual": 0.20, "attributes": 0.35, "fade_detail": 0.45}

        final_score = (
            effective_weights["visual"] * visual_score
            + effective_weights["attributes"] * attribute_score
            + effective_weights["fade_detail"] * fade_detail_score
            - 0.35 * mismatch_penalty
            - style_gate_penalty
        )
        final_score = min(score_cap, max(0.0, min(100.0, final_score)))
        score = int(round(final_score))

        style_gate = {
            "style_match_score": style_match,
            "back_length_match_score": back_length_match,
            "style_component_score": style_component,
            "back_component_score": back_component,
            "penalty": style_gate_penalty,
            "score_cap": score_cap,
            "reasons": style_gate_reasons,
            "fade_visibility": fade_visibility,
            "effective_weights": effective_weights,
        }

        fade_predictions = {
            group: {
                "reference": ref_fade_predictions[group],
                "result": res_fade_predictions[group],
            }
            for group in FADE_GROUPS
        }
        fade_confidences = {
            group: {
                "reference": ref_fade_confidence[group],
                "result": res_fade_confidence[group],
            }
            for group in FADE_GROUPS
        }

        fade_analysis: Dict[str, object] = {
            "score": fade_detail_score,
            "visual_similarity": fade_visual_similarity,
            "visual_score": fade_visual_score,
            "semantic_score": fade_semantic_score,
            "overlap_semantic_score": overlap_semantic_score,
            "discrete_semantic_score": discrete_semantic_score,
            "discrete_match_scores": {
                "fade_family": family_match,
                "fade_height": height_match,
                "fade_coverage": coverage_match,
                "base_length": base_match,
                "blend_quality": quality_match,
            },
            "visibility": {
                "reference": ref_visibility,
                "result": res_visibility,
            },
            "detail_similarities": fade_detail_similarities,
            "predictions": fade_predictions,
            "confidence": fade_confidences,
            "confidence_margin": {
                group: {
                    "reference": self._confidence_margin(ref_fade_profile[group]),
                    "result": self._confidence_margin(res_fade_profile[group]),
                }
                for group in FADE_GROUPS
            },
            "blend_quality_score": {
                "reference": reference_blend_quality,
                "result": result_blend_quality,
            },
            "mismatch_penalty": mismatch_penalty,
            "mismatch_reasons": mismatch_reasons,
        }

        return ScoreResult(
            score=score,
            raw_similarity=raw_similarity,
            view_similarities=view_similarities,
            component_scores={
                "visual": visual_score,
                "attributes": attribute_score,
                "fade_detail": fade_detail_score,
                "final": final_score,
            },
            attribute_similarities=attribute_similarities,
            attribute_predictions=attribute_predictions,
            style_gate=style_gate,
            fade_analysis=fade_analysis,
            device=self.device,
            model=f"{CLIP_MODEL}:{CLIP_PRETRAINED}:phase2f-haircut-family-gate",
        )
