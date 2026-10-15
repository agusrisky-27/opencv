#!/usr/bin/env python3
"""Shortcut runner agar 'python detect.py' bisa dijalankan langsung dari dalam folder sortir-yolo."""
import sys
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("detect_mod", HERE / "05_detect.py")
if spec and spec.loader:
    detect_mod = importlib.util.module_from_spec(spec)
    sys.modules["detect_mod"] = detect_mod
    spec.loader.exec_module(detect_mod)
    if __name__ == "__main__":
        sys.exit(detect_mod.main())
