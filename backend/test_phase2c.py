"""Quick local Phase 2C scorer test.

Usage:
    python test_phase2c.py "C:\\path\\reference.jpg" "C:\\path\\result.jpg"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ai.scorer import HaircutScorer


def main() -> int:
    if len(sys.argv) != 3:
        print('Usage: python test_phase2c.py "reference.jpg" "result.jpg"')
        return 2

    reference_path = Path(sys.argv[1])
    result_path = Path(sys.argv[2])

    if not reference_path.is_file() or not result_path.is_file():
        print("Both image paths must exist.")
        return 2

    scorer = HaircutScorer()
    result = scorer.compare(
        reference_path.read_bytes(),
        result_path.read_bytes(),
    )

    print(
        json.dumps(
            {
                "score": result.score,
                "raw_similarity": round(result.raw_similarity, 6),
                "view_similarities": {
                    k: round(v, 6) for k, v in result.view_similarities.items()
                },
                "component_scores": {
                    k: round(v, 2) for k, v in result.component_scores.items()
                },
                "attribute_similarities": {
                    k: round(v, 4)
                    for k, v in result.attribute_similarities.items()
                },
                "attribute_predictions": result.attribute_predictions,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
