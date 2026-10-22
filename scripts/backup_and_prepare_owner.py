"""
Secure, verified backup, preparation, and restore utility for Face Lock AI enrollment data.

Provides:
- Inventory of existing biometric images (data/) and encoding templates (encodings/)
- SHA-256 and byte-size verification before and after backup
- Full restorable archive in backups/
- Safe owner image clearing after verified backup (leaves other users and templates untouched)
- Restore utility to roll back templates and images from verified backups
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ENCODINGS_DIR = PROJECT_ROOT / "encodings"
BACKUPS_DIR = PROJECT_ROOT / "backups"


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file in 64 KiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory_files(owner_only: bool = False, owner_name: str = "owner") -> dict[str, dict[str, Any]]:
    """
    Inventory active biometric files across encodings/ and data/.
    """
    inventory: dict[str, dict[str, Any]] = {}
    safe_owner = owner_name.strip().lower().replace(" ", "_")

    if ENCODINGS_DIR.is_dir():
        for pkl in sorted(ENCODINGS_DIR.glob("*.pkl")):
            if owner_only and pkl.stem.lower() != safe_owner:
                continue
            rel = str(pkl.relative_to(PROJECT_ROOT))
            inventory[rel] = {
                "size": pkl.stat().st_size,
                "sha256": sha256_file(pkl),
            }

    if DATA_DIR.is_dir():
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for img in sorted(DATA_DIR.rglob(ext)):
                if owner_only and img.parent.name.lower() != safe_owner:
                    continue
                rel = str(img.relative_to(PROJECT_ROOT))
                inventory[rel] = {
                    "size": img.stat().st_size,
                    "sha256": sha256_file(img),
                }

    return inventory


def create_verified_backup(owner_name: str = "owner") -> Path:
    """
    Create a timestamped backup of biometric data and verify file counts, sizes, and hashes.
    Raises RuntimeError if verification fails.
    """
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"backup_{timestamp}"
    backup_dir = BACKUPS_DIR / backup_name
    backup_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Starting biometric inventory for backup...")
    source_inventory = inventory_files()
    print(f"[*] Found {len(source_inventory)} biometric files across data/ and encodings/.")

    if not source_inventory:
        print("[!] No biometric files found to back up.")
        return backup_dir

    # Copy files
    for rel_path, meta in source_inventory.items():
        src_file = PROJECT_ROOT / rel_path
        dst_file = backup_dir / rel_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)

    # Verification phase: check count, file existence, size, and SHA-256 hash
    print(f"[*] Verifying copied files against sizes and SHA-256 hashes...")
    verified_count = 0
    for rel_path, meta in source_inventory.items():
        dst_file = backup_dir / rel_path
        if not dst_file.is_file():
            raise RuntimeError(f"Backup verification failed: missing file {dst_file}")
        
        dst_size = dst_file.stat().st_size
        if dst_size != meta["size"]:
            raise RuntimeError(
                f"Backup verification failed: size mismatch for {dst_file}.\n"
                f"Source size: {meta['size']} bytes | Backup size: {dst_size} bytes"
            )

        dst_hash = sha256_file(dst_file)
        if dst_hash != meta["sha256"]:
            raise RuntimeError(
                f"Backup verification failed: SHA-256 mismatch for {dst_file}.\n"
                f"Source: {meta['sha256']}\nTarget: {dst_hash}"
            )
        verified_count += 1

    if verified_count != len(source_inventory):
        raise RuntimeError(
            f"Backup verification failed: expected {len(source_inventory)} files, verified {verified_count}"
        )

    # Save manifest
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "project_root": str(PROJECT_ROOT),
        "total_files": verified_count,
        "owner_name": owner_name,
        "files": source_inventory,
    }
    manifest_path = backup_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[+] Backup successfully verified: {verified_count}/{len(source_inventory)} files matched byte-for-byte.")
    print(f"[+] Backup stored at: {backup_dir}")
    return backup_dir


def verify_backup(backup_dir: Path) -> tuple[bool, str]:
    """
    Validate that an existing backup directory is intact against its manifest.
    """
    if not backup_dir.is_dir():
        return False, f"Backup directory does not exist: {backup_dir}"

    manifest_path = backup_dir / "manifest.json"
    if not manifest_path.is_file():
        return False, f"Missing manifest.json in backup: {backup_dir}"

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as err:
        return False, f"Corrupt manifest.json: {err}"

    files_dict = manifest.get("files", {})
    verified = 0
    for rel_path, meta in files_dict.items():
        dst_file = backup_dir / rel_path
        if not dst_file.is_file():
            return False, f"Missing backed-up file: {rel_path}"

        if dst_file.stat().st_size != meta.get("size"):
            return False, f"Size mismatch for {rel_path}"

        h = sha256_file(dst_file)
        if h != meta.get("sha256"):
            return False, f"Hash mismatch for {rel_path}"
        verified += 1

    return True, f"Verified {verified}/{len(files_dict)} files in {backup_dir.name}"


def prepare_clean_owner_environment(owner_name: str = "owner", backup_dir: Path | None = None) -> None:
    """
    Clear active enrollment images belonging to the configured owner ONLY after verified backup.
    Preserves other users' data, encodings, and all non-owner files.
    """
    if backup_dir is not None:
        valid, msg = verify_backup(backup_dir)
        if not valid:
            raise RuntimeError(f"Cannot clean owner environment: backup verification failed ({msg})")

    safe_owner = owner_name.strip().lower().replace(" ", "_")
    owner_data_folder = DATA_DIR / safe_owner

    if owner_data_folder.is_dir():
        print(f"[*] Clearing active enrollment images for owner '{safe_owner}'...")
        removed_count = 0
        for img in list(owner_data_folder.glob("*.*")):
            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                try:
                    img.unlink()
                    removed_count += 1
                except OSError as err:
                    print(f"[!] Warning: failed to remove {img.name}: {err}")
        print(f"[+] Cleared {removed_count} old images from {owner_data_folder}. Ready for clean 20-image capture.")
    else:
        owner_data_folder.mkdir(parents=True, exist_ok=True)
        print(f"[+] Initialized fresh owner image directory: {owner_data_folder}")


def restore_backup(backup_dir: Path | None = None, owner_name: str = "owner") -> bool:
    """
    Restore owner template and images from a verified backup.
    """
    if backup_dir is None:
        if not BACKUPS_DIR.is_dir():
            print("[!] No backups directory found.")
            return False
        backups = sorted([d for d in BACKUPS_DIR.iterdir() if d.is_dir() and (d / "manifest.json").is_file()])
        if not backups:
            print("[!] No verified backups found in backups/.")
            return False
        backup_dir = backups[-1]

    print(f"[*] Restoring from backup: {backup_dir.name}...")
    valid, msg = verify_backup(backup_dir)
    if not valid:
        print(f"[!] Backup verification failed: {msg}. Aborting restore.")
        return False

    manifest = json.loads((backup_dir / "manifest.json").read_text(encoding="utf-8"))
    files_dict = manifest.get("files", {})

    restored = 0
    for rel_path in files_dict:
        src = backup_dir / rel_path
        dst = PROJECT_ROOT / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        restored += 1

    print(f"[+] Successfully restored {restored} files from {backup_dir.name}.")
    return True


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "backup"
    if action == "backup":
        bkp = create_verified_backup()
        print(f"Backup created at: {bkp}")
    elif action == "verify":
        target = Path(sys.argv[2]) if len(sys.argv) > 2 else BACKUPS_DIR
        valid, msg = verify_backup(target)
        print(f"Verification: {valid} - {msg}")
    elif action == "restore":
        target = Path(sys.argv[2]) if len(sys.argv) > 2 else None
        restore_backup(target)
    else:
        print(f"Unknown action: {action}. Use 'backup', 'verify', or 'restore'.")
