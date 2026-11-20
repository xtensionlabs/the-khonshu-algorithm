import math

import numpy as np
import pytest

from khonshu import Cone, ConeColor, LocalPlanner, PlannerConfig, VehicleState


class StubBackend:
    def __init__(self, path=None, sorted_left=None, sorted_right=None):
        self.path = np.asarray(
            path
            if path is not None
            else [[0.0, 1.0, 0.0, 0.0], [1.0, 2.0, 0.0, 0.0]],
            dtype=float,
        )
        self.sorted_left = np.asarray(sorted_left if sorted_left is not None else [[1, 2], [2, 2]])
        self.sorted_right = np.asarray(
            sorted_right if sorted_right is not None else np.empty((0, 2))
        )
        self.calls = []

    def calculate_path_in_global_frame(self, cones, position, direction, return_intermediate_results):
        self.calls.append((cones, position, direction, return_intermediate_results))
        return self.path, self.sorted_left, self.sorted_right, None, None, None, None


def sample_cones():
    return [
        Cone(3.0, 2.0, ConeColor.BLUE),
        Cone(6.0, 2.0, ConeColor.BLUE),
        Cone(3.0, -2.0, ConeColor.YELLOW),
    ]


def sample_state(yaw=0.0):
    return VehicleState(0.0, 0.0, yaw, 0.0, 12.5, frame_id="map")


@pytest.mark.parametrize("cone_count", [0, 1, 2])
def test_insufficient_cones_request_stop_without_calling_backend(cone_count):
    backend = StubBackend()
    result = LocalPlanner(backend=backend).plan(sample_cones()[:cone_count], sample_state())

    assert result.emergency_stop
    assert not result.valid
    assert result.points == ()
    assert result.reason == "insufficient_cones"
    assert backend.calls == []


@pytest.mark.parametrize(
    ("yaw", "expected_direction"),
    [
        (0.0, (1.0, 0.0)),
        (math.pi / 2.0, (0.0, 1.0)),
        (-math.pi / 2.0, (0.0, -1.0)),
    ],
)
def test_state_yaw_is_converted_to_forward_unit_vector(yaw, expected_direction):
    backend = StubBackend()
    result = LocalPlanner(backend=backend).plan(sample_cones(), sample_state(yaw))

    assert result.valid
    np.testing.assert_allclose(backend.calls[0][2], expected_direction, atol=1e-12)
    assert backend.calls[0][3] is True


@pytest.mark.parametrize(
    ("path", "reason"),
    [
        ([[0.0, 1.0, 0.0, 0.0]], "invalid_path_shape"),
        ([[0.0, 1.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]], "non_monotonic_path_distance"),
        ([[0.0, 50.0, 0.0, 0.0], [1.0, 51.0, 0.0, 0.0]], "path_too_far_from_vehicle"),
        ([[0.0, 1.0, 0.0, 0.0], [1.0, float("nan"), 0.0, 0.0]], "non_finite_path"),
    ],
)
def test_untrusted_path_requests_stop(path, reason):
    backend = StubBackend(path=path)
    result = LocalPlanner(backend=backend).plan(sample_cones(), sample_state())

    assert result.emergency_stop
    assert not result.points
    assert result.reason == reason


def test_missing_sorted_boundary_evidence_requests_stop():
    backend = StubBackend(sorted_left=[[1.0, 2.0]])
    result = LocalPlanner(backend=backend).plan(sample_cones(), sample_state())

    assert result.emergency_stop
    assert result.reason == "insufficient_boundary_evidence"


def test_malformed_sorted_boundary_requests_stop():
    backend = StubBackend(sorted_left=[1.0, 2.0])
    result = LocalPlanner(backend=backend).plan(sample_cones(), sample_state())

    assert result.emergency_stop
    assert result.reason == "invalid_boundary_shape"


def test_path_too_close_to_observed_cone_requests_stop():
    backend = StubBackend(
        path=[[0.0, 3.0, 2.0, 0.0], [1.0, 4.0, 2.0, 0.0]]
    )
    result = LocalPlanner(backend=backend).plan(sample_cones(), sample_state())

    assert result.emergency_stop
    assert result.reason == "cone_clearance_violation"


def test_cycle_over_budget_requests_stop():
    backend = StubBackend()
    planner = LocalPlanner(PlannerConfig(maximum_cycle_time_ms=1e-12), backend=backend)

    result = planner.plan(sample_cones(), sample_state())

    assert result.emergency_stop
    assert result.reason == "planning_deadline_exceeded"


def test_valid_path_has_constant_safe_speed_and_preserves_frame_and_stamp():
    backend = StubBackend(
        path=[[0.0, 1.0, 0.1, 0.01], [0.5, 1.5, 0.2, 0.02], [1.0, 2.0, 0.3, 0.03]]
    )
    result = LocalPlanner(PlannerConfig(safe_speed_mps=1.75), backend=backend).plan(
        sample_cones(), sample_state()
    )

    assert result.valid
    assert not result.emergency_stop
    assert result.frame_id == "map"
    assert result.timestamp_s == 12.5
    assert [point.target_speed_mps for point in result.points] == [1.75] * 3
    assert [point.s_m for point in result.points] == [0.0, 0.5, 1.0]


@pytest.mark.parametrize(
    "state_values",
    [
        (float("nan"), 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, -0.1, 0.0),
        (0.0, 0.0, 0.0, 0.0, float("inf")),
    ],
)
def test_vehicle_state_rejects_invalid_values(state_values):
    with pytest.raises(ValueError):
        VehicleState(*state_values)
