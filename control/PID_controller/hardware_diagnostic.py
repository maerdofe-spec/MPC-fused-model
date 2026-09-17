"""Disarmed-only serial diagnostic for V4pro1_MPC firmware v5."""

from __future__ import annotations

import argparse
import time

try:
    from .finesub_transport import (
        FineSUBConnection,
        SerialTransport,
        load_runtime_config,
        make_transport,
    )
except ImportError:
    from finesub_transport import (
        FineSUBConnection,
        SerialTransport,
        load_runtime_config,
        make_transport,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify FineSUB v5 telemetry and handshake without arming motors"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--port", help="direct USART bridge, for example /dev/ttyUSB0")
    source.add_argument(
        "--runtime-config",
        help="MPC-style JSON; uses its configured TCP/UDP/serial transport",
    )
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.seconds <= 0.0:
        parser.error("--seconds must be positive")

    if args.runtime_config:
        config = load_runtime_config(args.runtime_config)
        transport_config = config.get("transport")
        if not isinstance(transport_config, dict):
            parser.error("runtime config must contain a transport object")
        control_config = config.get("control")
        if not isinstance(control_config, dict):
            control_config = {}
        transport = make_transport(transport_config)
        telemetry_max_age_s = float(control_config.get("telemetry_max_age_sec", 0.20))
        confirmation_max_age_s = float(
            control_config.get("confirmation_max_age_sec", 0.30)
        )
        reconnect_interval_s = float(
            transport_config.get("reconnect_interval_sec", 1.0)
        )
    else:
        transport = SerialTransport(args.port, baudrate=115200)
        telemetry_max_age_s = 0.20
        confirmation_max_age_s = 0.30
        reconnect_interval_s = 1.0
    link = FineSUBConnection(
        transport,
        telemetry_max_age_s=telemetry_max_age_s,
        confirmation_max_age_s=confirmation_max_age_s,
        reconnect_interval_s=reconnect_interval_s,
        logger=print,
    )
    if not link.connect():
        return 2
    deadline = time.monotonic() + args.seconds
    next_send = 0.0
    next_print = 0.0
    telemetry_seen = False
    session_confirmation_seen = False
    try:
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_send:
                link.send_disarm()  # This script never sets the armed flag.
                next_send = now + 0.05
            telemetry = link.fresh_telemetry()
            if telemetry is not None and now >= next_print:
                telemetry_seen = True
                session_confirmation_seen = session_confirmation_seen or link.confirmation_fresh()
                print(
                    "telemetry",
                    f"seq={telemetry.sequence}",
                    f"session_ok={link.confirmation_fresh()}",
                    f"armed={telemetry.armed}",
                    f"failsafe={telemetry.failsafe}",
                    f"yaw={telemetry.yaw_rad:.4f}rad",
                    f"yaw_rate={telemetry.yaw_rate_rad_s:.4f}rad/s",
                    f"depth={telemetry.depth_m:.3f}m",
                    f"rpm_mask=0x{telemetry.rpm_valid_mask:02x}",
                )
                next_print = now + 0.5
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            for _ in range(3):
                link.send_disarm()
                time.sleep(0.02)
        finally:
            link.close()
    # A serial port accepting bytes is not sufficient evidence that it is the
    # V4pro1 USART3 bridge. Require both fresh telemetry and an echoed command
    # confirmation before reporting a usable session; this script remains
    # disarm-only regardless of the result.
    return 0 if telemetry_seen and session_confirmation_seen and link.decoder.crc_errors == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
