#!/usr/bin/env python3
"""Stable command-line entry for the formal FineSUB AUTO-only runtime."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "control"))

from MPC_dual_model.auto_only_runtime import main


if __name__ == "__main__":
    raise SystemExit(main())

