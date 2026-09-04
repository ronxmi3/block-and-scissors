"""Phase 2F FastAPI backend with haircut-family gating and fade precision diagnostics."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from ai.scorer import HaircutScorer
from blockchain.sui_oracle import check_sui_environment, resolve_escrow
from config import MAX_IMAGE_BYTES, ORACLE_API_KEY, SUI_THRESHOLD


app = FastAPI(
    title="AI Haircut Escrow Oracle",
    version="0.6.0-phase2f-haircut-family-gate",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_scorer() -> HaircutScorer:
    return HaircutScorer()


async def read_image(upload: UploadFile) -> bytes:
    if not upload.content_type or not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"{upload.filename!r} is not an image upload.")

    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail=f"{upload.filename!r} is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{upload.filename!r} is larger than the {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit.",
        )
    return data


def verify_oracle_key(received_key: str | None) -> None:
    if not received_key or received_key != ORACLE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Oracle-Key.")


def _rounded_nested(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: _rounded_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded_nested(item) for item in value]
    return value


def _score_payload(score_result) -> dict:
    return {
        "score": score_result.score,
        "raw_similarity": round(score_result.raw_similarity, 6),
        "view_similarities": {
            key: round(value, 6) for key, value in score_result.view_similarities.items()
        },
        "component_scores": {
            key: round(value, 2) for key, value in score_result.component_scores.items()
        },
        "attribute_similarities": {
            key: round(value, 4) for key, value in score_result.attribute_similarities.items()
        },
        "attribute_predictions": score_result.attribute_predictions,
        "style_gate": _rounded_nested(score_result.style_gate),
        "fade_analysis": _rounded_nested(score_result.fade_analysis),
        "threshold": SUI_THRESHOLD,
        "predicted_outcome": (
            "BARBER_PAID" if score_result.score >= SUI_THRESHOLD else "CUSTOMER_REFUNDED"
        ),
        "model": score_result.model,
        "device": score_result.device,
        "warning": (
            "Phase 2F is an MVP haircut/fade similarity score, not a probability, "
            "craftsmanship guarantee, or fairness guarantee. It adds a haircut-family gate so "
            "generic visual similarity cannot override a major style/length mismatch, but it still "
            "requires calibration on a diverse labelled test set before real-money use."
        ),
    }


@app.get("/health")
def health() -> dict:
    try:
        sui = check_sui_environment()
        sui_ok = True
        sui_error = None
    except Exception as exc:
        sui = None
        sui_ok = False
        sui_error = str(exc)

    return {
        "backend": "ok",
        "ai_version": "phase2f-haircut-family-gate",
        "sui_ok": sui_ok,
        "sui": sui,
        "sui_error": sui_error,
        "threshold": SUI_THRESHOLD,
    }


@app.post("/score")
async def score_haircut(
    reference: UploadFile = File(...),
    result: UploadFile = File(...),
) -> dict:
    reference_bytes = await read_image(reference)
    result_bytes = await read_image(result)

    try:
        score_result = get_scorer().compare(reference_bytes, result_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not score images: {exc}") from exc

    return _score_payload(score_result)


@app.post("/evaluate-and-resolve")
async def evaluate_and_resolve(
    escrow_id: str = Form(...),
    reference: UploadFile = File(...),
    result: UploadFile = File(...),
    dry_run: bool = Form(True),
    x_oracle_key: str | None = Header(default=None),
) -> dict:
    verify_oracle_key(x_oracle_key)

    reference_bytes = await read_image(reference)
    result_bytes = await read_image(result)

    try:
        score_result = get_scorer().compare(reference_bytes, result_bytes)
        sui_result = resolve_escrow(
            escrow_id=escrow_id,
            score=score_result.score,
            dry_run=dry_run,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    payload = _score_payload(score_result)
    payload["sui"] = {
        "success": sui_result.success,
        "dry_run": sui_result.dry_run,
        "transaction_digest": sui_result.transaction_digest,
    }
    return payload
