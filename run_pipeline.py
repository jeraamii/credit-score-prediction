"""
run_pipeline.py
Entrypoint — jalankan file ini untuk melatih semua model.

Usage:
    python run_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src" / "pipeline"))
from pipeline import run_pipeline  # noqa: E402

if __name__ == "__main__":
    run_pipeline()
