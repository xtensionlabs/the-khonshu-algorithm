import pytest

from khonshu import LocalPlanner
from khonshu.scenarios import synthetic_scenarios


@pytest.mark.integration
@pytest.mark.parametrize("scenario_name", tuple(synthetic_scenarios()))
def test_vendored_planner_handles_shared_synthetic_scenarios(scenario_name):
    planner = LocalPlanner()
    planner.warm_up()
    cones, state = synthetic_scenarios()[scenario_name]

    trajectory = planner.plan(cones, state)

    assert trajectory.valid, f"{scenario_name}: {trajectory.reason}"
    assert len(trajectory.points) >= 2
    assert all(point.target_speed_mps == 2.0 for point in trajectory.points)
