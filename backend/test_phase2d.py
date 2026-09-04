"""Quick local Phase 2D scorer smoke test.

Usage from backend folder:
    python test_phase2d.py reference.jpg result.jpg
"""

import json
import sys
from pathlib import Path

from ai.scorer import HaircutScorer


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python test_phase2d.py reference.jpg result.jpg")

    reference = Path(sys.argv[1]).read_bytes()
    result = Path(sys.argv[2]).read_bytes()

    scored = HaircutScorer().compare(reference, result)
    print(json.dumps({
        "score": scored.score,
        "component_scores": scored.component_scores,
        "attribute_predictions": scored.attribute_predictions,
        "fade_analysis": scored.fade_analysis,
        "model": scored.model,
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
