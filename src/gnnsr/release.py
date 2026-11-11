"""Verify this public source snapshot without depending on a historical deployment."""
import hashlib
import json
from pathlib import Path


def verify_source(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or invalid source path: {record['path']}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Source differs from release manifest: {record['path']}")
    return len(manifest["files"])

