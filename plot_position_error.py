#!/usr/bin/env python3
"""Stable launcher for the live FineSUB MPC position-error plot."""

import os

# Matplotlib otherwise selects the non-interactive Agg backend in this uv
# environment even when the pool-side desktop is available.
os.environ.setdefault("MPLBACKEND", "QtAgg" if os.environ.get("DISPLAY") else "Agg")

from MPC_dual_model.realtime_position_error_plot import main


if __name__ == "__main__":
    raise SystemExit(main())
