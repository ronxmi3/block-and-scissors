"""
Command-line test for Phase 2A.

Usage:
    python test_score.py reference.jpg after.jpg
"""

import sys
from pathlib import Path

from ai.scorer import HaircutScorer


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage:")
        print("    python test_score.py reference.jpg after.jpg")
        raise SystemExit(1)

    reference_path = Path(sys.argv[1])
    result_path = Path(sys.argv[2])

    if not reference_path.exists():
        raise FileNotFoundError(reference_path)

    if not result_path.exists():
        raise FileNotFoundError(result_path)

    scorer = HaircutScorer()

    result = scorer.compare(
        reference_path.read_bytes(),
        result_path.read_bytes(),
    )

    print("\n==============================")
    print("HAIRCUT SIMILARITY RESULT")
    print("==============================")
    print(f"Score:          {result.score}/100")
    print(f"Raw similarity: {result.raw_similarity:.4f}")
    print(f"Model:          {result.model}")
    print(f"Device:         {result.device}")

    print("\nPer-view cosine similarity:")
    for name, value in result.view_similarities.items():
        print(f"  {name:16s}: {value:.4f}")


if __name__ == "__main__":
    main()
