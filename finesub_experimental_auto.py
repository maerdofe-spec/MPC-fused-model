#!/usr/bin/env python3
"""Stable entry point for the explicitly risk-accepted FineSUB experiment."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "control"))

from MPC_dual_model.experimental_auto import main


if __name__ == "__main__":
    raise SystemExit(main())
