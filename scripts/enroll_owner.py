"""
Dedicated owner enrollment CLI for Face Lock AI.

Runs guided 20-image owner enrollment using the existing registration pipeline,
pretrained FaceNet512 embeddings, verified backup protection, and atomic HMAC-SHA256
signed template replacement.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backup_and_prepare_owner import (
    create_verified_backup,
    prepare_clean_owner_environment,
    verify_backup,
)
from src.config import RECOGNITION_SETTINGS
from src.operator_auth import verify_operator_pin
from src.register import register_user
from src.utils import print_status


def enroll_owner(overwrite: bool = True, skip_auth: bool = False) -> int:
    """
    Run guided 20-image enrollment for the configured owner.
    """
    owner_name = RECOGNITION_SETTINGS.owner_name

    print_status(f"Starting guided 20-image owner enrollment for: '{owner_name}'", level="info")
    print_status("Enrollment Specifications:", level="info")
    print_status("  1. Exactly 20 accepted, distinct face images will be captured automatically.", level="info")
    print_status("  2. Rejects blur, poor lighting, clipping, extreme motion, and duplicate frames.", level="info")
    print_status("  3. Follow on-screen pose guidance for natural head angles and subtle expressions.", level="info")
    print_status("  4. Presentation Attack Detection (PAD) liveness check is enforced.", level="info")
    print_status("  5. Generates 20 valid 512-D FaceNet512 embeddings signed with HMAC-SHA256.", level="info")
    print_status("  6. Atomic replacement: previous template is preserved on cancellation or failure.", level="info")

    # Operator authentication gate
    if not skip_auth and not verify_operator_pin():
        print_status("Enrollment denied: operator authentication failed.", level="error")
        return 1

    # Step 1: Data protection & verified backup
    print_status("Step 1: Creating verified backup of existing enrollment data...", level="info")
    backup_dir: Path | None = None
    try:
        backup_dir = create_verified_backup(owner_name=owner_name)
        valid, msg = verify_backup(backup_dir)
        if not valid:
            print_status(f"Backup verification failed: {msg}. Aborting enrollment.", level="error")
            return 1
        print_status(f"Backup verified ({msg}): {backup_dir}", level="success")
    except Exception as err:
        print_status(f"Backup error: {err}. Stopping without modifying any files.", level="error")
        return 1

    # Step 2: Clear active enrollment images belonging to the configured owner
    print_status("Step 2: Preparing clean owner enrollment environment...", level="info")
    try:
        prepare_clean_owner_environment(owner_name=owner_name, backup_dir=backup_dir)
    except Exception as err:
        print_status(f"Preparation error: {err}. Aborting enrollment.", level="error")
        return 1

    # Step 3: Guided 20-image capture and atomic template generation
    print_status("Step 3: Launching guided 20-image camera capture...", level="info")
    result = register_user(
        user_name=owner_name,
        image_count=20,
        minimum_embeddings=20,
        overwrite=overwrite,
    )

    if result.get("success"):
        print_status(
            f"Owner enrollment SUCCESSFUL: '{owner_name}' enrolled with "
            f"{result['embedding_count']} valid 512-D embeddings.",
            level="success",
        )
        print_status(f"Template path: {result['encoding_path']}", level="success")
        print_status(f"Previous enrollment backed up at: {backup_dir}", level="info")
        return 0

    reason = result.get("reason", "unknown_error")
    if reason == "cancelled":
        print_status("Owner enrollment cancelled by user.", level="warning")
    else:
        print_status(f"Owner enrollment not completed: {reason}", level="error")

    print_status("Previous template remained intact and was not overwritten.", level="info")
    if backup_dir is not None:
        print_status(f"Backup is safely preserved at: {backup_dir}", level="info")
        print_status(
            f"To restore previous template and images if needed, run:\n"
            f"  python scripts/backup_and_prepare_owner.py restore {backup_dir}",
            level="info",
        )
    return 1 if reason != "cancelled" else 0


if __name__ == "__main__":
    sys.exit(enroll_owner(overwrite=True))
