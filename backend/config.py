"""
Central configuration for Phase 2.

All values are read from `.env`, so package IDs and secrets are not
scattered throughout the backend source code.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


# Load backend/.env when this module is imported.
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


SUI_PACKAGE_ID = os.getenv(
    "SUI_PACKAGE_ID",
    "0x5d7fa930a5d95ae7f8a2a56693e5341cb8e84dd0865fa42a6aac215c2659057a",
)

SUI_ORACLE_CAP_ID = os.getenv(
    "SUI_ORACLE_CAP_ID",
    "0x938048aa8ae82f694635f5723e54c3cc916554961c0b815df69556bd1a94caac",
)

SUI_THRESHOLD = int(os.getenv("SUI_THRESHOLD", "80"))
SUI_GAS_BUDGET_MIST = int(os.getenv("SUI_GAS_BUDGET_MIST", "50000000"))

# Temporary protection for the oracle endpoint.
# Do NOT expose the settlement endpoint publicly with a weak/default key.
ORACLE_API_KEY = os.getenv("ORACLE_API_KEY", "change-me")

CLIP_MODEL = os.getenv("CLIP_MODEL", "ViT-B-32")
CLIP_PRETRAINED = os.getenv("CLIP_PRETRAINED", "laion2b_s34b_b79k")

# Provisional score calibration.
SIMILARITY_FLOOR = float(os.getenv("SIMILARITY_FLOOR", "0.50"))
SIMILARITY_CEILING = float(os.getenv("SIMILARITY_CEILING", "0.88"))

# Upload safety limit. Haircut photos do not need to be enormous.
MAX_IMAGE_BYTES = 10 * 1024 * 1024
