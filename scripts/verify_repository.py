"""Verify source checksums before installing dependencies (standard library only)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnnsr.release import verify_source

if __name__ == "__main__":
    print(f"Verified {verify_source(ROOT)} source files.")

