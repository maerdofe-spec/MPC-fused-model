import numpy as np

from MPC_dual_model.realtime_position_error_plot import (
    extract_position_error_sample,
)


def test_extracts_measured_and_estimated_position_error() -> None:
    record = {
        "event": "control_update",
        "host_monotonic_s": 12.5,
        "position_body_frd_m": [0.90, -0.02, -0.10],
        "estimated_state": [0.88, -0.03, -0.11, 1.0, 2.0, 3.0],
        "model1_weight": [0.9, 0.8, 0.7],
        "reference_position_body_frd_m": None,
    }
    sample = extract_position_error_sample(
        record,
        np.asarray([0.85, -0.05, -0.12]),
    )
    assert sample is not None
    np.testing.assert_allclose(sample.measured_error_m, [0.05, 0.03, 0.02])
    np.testing.assert_allclose(sample.estimated_error_m, [0.03, 0.02, 0.01])
    np.testing.assert_allclose(sample.estimated_velocity_m_s, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(sample.model1_weight, [0.9, 0.8, 0.7])


def test_uses_per_trace_locked_reference() -> None:
    record = {
        "event": "control_update",
        "host_time_s": 20.0,
        "position_body_frd_m": [1.0, 2.0, 3.0],
        "estimated_state": [1.1, 2.2, 3.3],
        "reference_position_body_frd_m": [1.0, 2.0, 3.0],
    }
    sample = extract_position_error_sample(record, np.zeros(3))
    assert sample is not None
    np.testing.assert_allclose(sample.measured_error_m, np.zeros(3))
    np.testing.assert_allclose(sample.estimated_error_m, [0.1, 0.2, 0.3])


def test_ignores_non_control_events() -> None:
    assert extract_position_error_sample({"event": "start"}, np.zeros(3)) is None
