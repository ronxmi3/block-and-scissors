"""
Sui oracle bridge.

For Phase 2 we deliberately reuse the Sui CLI that is already installed,
configured for Testnet, and proven to work on this machine.

Later, the frontend/backend can move to a dedicated Sui SDK if desired.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

from config import (
    SUI_GAS_BUDGET_MIST,
    SUI_ORACLE_CAP_ID,
    SUI_PACKAGE_ID,
)


@dataclass
class SuiResolutionResult:
    success: bool
    transaction_digest: Optional[str]
    stdout: str
    stderr: str
    dry_run: bool


def _validate_object_id(value: str, field_name: str) -> str:
    """
    Reject obviously malformed IDs before passing them to the CLI.

    Sui object IDs are hexadecimal strings prefixed by 0x.
    """
    value = value.strip()

    if not re.fullmatch(r"0x[0-9a-fA-F]{1,64}", value):
        raise ValueError(f"Invalid {field_name}: {value}")

    return value


def check_sui_environment() -> dict:
    """
    Confirm that:
    - `sui` exists
    - active environment is Testnet
    - an active address exists
    """
    sui_path = shutil.which("sui")

    if not sui_path:
        raise RuntimeError(
            "Could not find the `sui` executable in PATH."
        )

    env_result = subprocess.run(
        [sui_path, "client", "active-env"],
        capture_output=True,
        text=True,
        check=False,
    )

    if env_result.returncode != 0:
        raise RuntimeError(env_result.stderr or env_result.stdout)

    active_env = env_result.stdout.strip()

    if active_env.lower() != "testnet":
        raise RuntimeError(
            f"Expected active Sui environment 'testnet', got '{active_env}'."
        )

    address_result = subprocess.run(
        [sui_path, "client", "active-address"],
        capture_output=True,
        text=True,
        check=False,
    )

    if address_result.returncode != 0:
        raise RuntimeError(address_result.stderr or address_result.stdout)

    return {
        "sui_path": sui_path,
        "active_env": active_env,
        "active_address": address_result.stdout.strip(),
    }


def resolve_escrow(
    escrow_id: str,
    score: int,
    *,
    dry_run: bool = True,
) -> SuiResolutionResult:
    """
    Submit the AI score to:
        haircut_escrow::resolve_escrow

    This is the Python equivalent of the PowerShell command that already
    paid/refunded your test escrows.
    """
    if not 0 <= score <= 100:
        raise ValueError("score must be between 0 and 100")

    escrow_id = _validate_object_id(escrow_id, "escrow_id")
    oracle_cap = _validate_object_id(
        SUI_ORACLE_CAP_ID,
        "SUI_ORACLE_CAP_ID",
    )
    package_id = _validate_object_id(
        SUI_PACKAGE_ID,
        "SUI_PACKAGE_ID",
    )

    env = check_sui_environment()
    sui_path = env["sui_path"]

    move_function = (
        f"{package_id}::haircut_escrow::resolve_escrow"
    )

    command = [
        sui_path,
        "client",
        "ptb",
        "--move-call",
        move_function,
        f"@{escrow_id}",
        f"@{oracle_cap}",
        f"{score}u8",
        "--gas-budget",
        str(SUI_GAS_BUDGET_MIST),
    ]

    if dry_run:
        command.append("--dry-run")

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    combined_output = f"{result.stdout}\n{result.stderr}"

    digest_match = re.search(
        r"(?:Transaction Digest|Digest):\s*([A-Za-z0-9]+)",
        combined_output,
    )

    transaction_digest = (
        digest_match.group(1)
        if digest_match
        else None
    )

    success = (
        result.returncode == 0
        and (
            "Status: Success" in combined_output
            or "execution status: success" in combined_output.lower()
        )
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Sui transaction failed to build/execute.\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )

    return SuiResolutionResult(
        success=success,
        transaction_digest=transaction_digest,
        stdout=result.stdout,
        stderr=result.stderr,
        dry_run=dry_run,
    )
