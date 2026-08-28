import math
import warnings
import mini_ode

import pytest

from akson._utils import _round_to_nice, resolve_default_solver


def test_round_to_nice_zero_returns_zero():
    assert _round_to_nice(0.0) == 0.0


def test_round_to_nice_negative_returns_unchanged():
    assert _round_to_nice(-5.0) == -5.0


def test_round_to_nice_exact_power_of_ten():
    assert _round_to_nice(1.0) == pytest.approx(1.0)


def test_round_to_nice_snaps_down_to_five():
    assert _round_to_nice(7.0) == pytest.approx(
        5.0
    )  # mantissa 7 -> largest candidate <= 7 is 5


def test_round_to_nice_snaps_to_two_point_five():
    assert _round_to_nice(0.03) == pytest.approx(0.025)  # mantissa 3.0 -> 2.5


def test_round_to_nice_exact_two_point_five():
    assert _round_to_nice(25.0) == pytest.approx(25.0)


def test_round_to_nice_snaps_to_two():
    assert _round_to_nice(2.4) == pytest.approx(2.0)  # mantissa 2.4 -> 2


def test_round_to_nice_never_exceeds_input():
    for value in [0.137, 1.0, 13.7, 99.9, 1234.5]:
        assert _round_to_nice(value) <= value


# Duck-typed solver stand-ins, so these tests don't depend on the exact
# numerical behaviour of mini_ode's real solver implementations.
class _FakeFixedStepSolver:
    def __init__(self, step, stability_radius=float("inf")):
        self.step = step
        self.stability_radius = stability_radius


class _FakeAdaptiveSolver:
    def __init__(self, stability_radius=float("inf")):
        self.stability_radius = stability_radius


class _FakeSolverWithoutStep:
    pass


def test_resolve_default_solver_rejects_zero_duration():
    with pytest.raises(ValueError, match="Duration must be positive"):
        resolve_default_solver(0.0, None, None, None)


def test_resolve_default_solver_rejects_negative_duration():
    with pytest.raises(ValueError, match="Duration must be positive"):
        resolve_default_solver(-1.0, None, None, None)


def test_resolve_default_solver_rejects_nonpositive_step_size():
    with pytest.raises(ValueError, match="Step size must be positive"):
        resolve_default_solver(10.0, -0.5, None, None)


def test_resolve_default_solver_rejects_nonpositive_spectral_radius():
    with pytest.raises(ValueError, match="Spectral radius must be positive"):
        resolve_default_solver(10.0, None, None, max_spectral_radius=0.0)


def test_resolve_default_solver_fixed_step_solver_missing_step_attribute_raises():
    solver = _FakeSolverWithoutStep()
    with pytest.raises(ValueError, match="Fixed-step solver must expose"):
        resolve_default_solver(10.0, None, solver, None, require_fixed_step=True)


def test_resolve_default_solver_fixed_step_solver_step_takes_precedence():
    solver = _FakeFixedStepSolver(step=0.25)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual_step, resolved_solver = resolve_default_solver(
            10.0, 0.5, solver, None, require_fixed_step=True
        )
    assert actual_step == 0.25
    assert resolved_solver is solver


def test_resolve_default_solver_fixed_step_warns_when_solver_and_step_size_given():
    solver = _FakeFixedStepSolver(step=0.25)
    with pytest.warns(
        UserWarning, match="solver's own 'step' attribute takes precedence"
    ):
        resolve_default_solver(10.0, 0.5, solver, None, require_fixed_step=True)


def test_resolve_default_solver_fixed_step_uses_step_size_when_no_solver():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual_step, resolved_solver = resolve_default_solver(
            10.0, 0.3, None, None, require_fixed_step=True
        )
    assert actual_step == 0.3
    assert resolved_solver.step == 0.3


def test_resolve_default_solver_fixed_step_uses_stability_recommendation():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual_step, _ = resolve_default_solver(
            100.0, None, None, max_spectral_radius=2.0, require_fixed_step=True
        )
    assert actual_step > 0
    assert math.isfinite(actual_step)


def test_resolve_default_solver_fixed_step_falls_back_to_duration_over_100():
    with pytest.warns(RuntimeWarning, match="max_spectral_radius is unknown"):
        actual_step, _ = resolve_default_solver(
            100.0, None, None, max_spectral_radius=None, require_fixed_step=True
        )
    assert actual_step == pytest.approx(1.0)


def test_resolve_default_solver_adaptive_solver_and_step_are_independent():
    solver = _FakeAdaptiveSolver()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual_step, resolved_solver = resolve_default_solver(
            10.0, 0.7, solver, None, require_fixed_step=False
        )
    assert actual_step == 0.7
    assert resolved_solver is solver


def test_resolve_default_solver_adaptive_defaults_to_duration_over_100():
    with pytest.warns(RuntimeWarning, match="max_spectral_radius is unknown"):
        actual_step, _ = resolve_default_solver(
            50.0,
            None,
            _FakeAdaptiveSolver(),
            max_spectral_radius=None,
            require_fixed_step=False,
        )
    assert actual_step == pytest.approx(0.5)


def test_resolve_default_solver_adaptive_creates_default_solver_when_none_given():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual_step, resolved_solver = resolve_default_solver(
            10.0, 0.1, None, None, require_fixed_step=False
        )
    assert resolved_solver is not None
    assert actual_step == 0.1


def test_resolve_default_solver_discrete_snaps_recommended_step():
    # duration/100 = 1.37 -> snapped down to the nice value 1.0
    with pytest.warns(RuntimeWarning, match="max_spectral_radius is unknown"):
        actual_step, _ = resolve_default_solver(
            137.0,
            None,
            None,
            max_spectral_radius=None,
            require_fixed_step=True,
            discrete=True,
        )
    assert actual_step == pytest.approx(1.0)


def test_resolve_default_solver_warns_when_step_exceeds_stability_limit():
    # max_stable_step = stability_radius / max_spectral_radius = 1.0 / 2.0 = 0.5,
    # and the solver's step (10.0) far exceeds it.
    solver = _FakeFixedStepSolver(step=10.0, stability_radius=1.0)
    with pytest.warns(RuntimeWarning, match="exceeds the stability limit"):
        resolve_default_solver(
            100.0, None, solver, max_spectral_radius=2.0, require_fixed_step=True
        )


def test_resolve_default_solver_warns_when_step_much_smaller_than_recommended():
    solver = _FakeFixedStepSolver(step=1e-6, stability_radius=1e6)
    with pytest.warns(RuntimeWarning, match="unnecessarily slow"):
        resolve_default_solver(
            1000.0, None, solver, max_spectral_radius=1.0, require_fixed_step=True
        )


def test_resolve_default_solver_no_stability_warning_when_step_size_reasonable():
    solver = _FakeFixedStepSolver(step=0.4, stability_radius=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        actual_step, _ = resolve_default_solver(
            100.0, None, solver, max_spectral_radius=2.0, require_fixed_step=True
        )
    assert actual_step == 0.4


def test_duration_must_be_positive():
    with pytest.raises(ValueError, match="Duration must be positive"):
        resolve_default_solver(
            duration=0.0,
            step_size=None,
            solver=None,
            max_spectral_radius=1.0,
            require_fixed_step=True,
        )


def test_step_size_must_be_positive():
    with pytest.raises(ValueError, match="Step size must be positive"):
        resolve_default_solver(
            duration=1.0,
            step_size=0.0,
            solver=None,
            max_spectral_radius=1.0,
            require_fixed_step=True,
        )


def test_spectral_radius_must_be_positive():
    with pytest.raises(ValueError, match="Spectral radius must be positive"):
        resolve_default_solver(
            duration=1.0,
            step_size=None,
            solver=None,
            max_spectral_radius=0.0,
            require_fixed_step=True,
        )


def test_fixed_step_solver_takes_precedence_over_step_size():
    solver = mini_ode.RK4MethodSolver(step=0.5)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        actual_step, resolved_solver = resolve_default_solver(
            duration=1.0,
            step_size=0.1,  # Should be ignored
            solver=solver,
            max_spectral_radius=1.0,
            require_fixed_step=True,
        )

        assert actual_step == 0.5
        assert resolved_solver is solver
        assert any("ignored" in str(warn.message).lower() for warn in w)


def test_fixed_step_uses_step_size_when_no_solver():
    actual_step, resolved_solver = resolve_default_solver(
        duration=1.0,
        step_size=0.1,
        solver=None,
        max_spectral_radius=1.0,
        require_fixed_step=True,
    )

    assert actual_step == 0.1
    assert resolved_solver.__repr__() == "RK4MethodSolver(step=0.1)"
    assert resolved_solver.step == 0.1


def test_fixed_step_uses_recommended_when_no_solver_no_step_size():
    # For max_spectral_radius=2.0 and RK4 (stability_radius ~= 2.8),
    # recommended = 2.8 / 2.0 / 2 = 0.7
    actual_step, resolved_solver = resolve_default_solver(
        duration=10.0,
        step_size=None,
        solver=None,
        max_spectral_radius=2.0,
        require_fixed_step=True,
    )

    assert actual_step > 0
    assert resolved_solver.__repr__()[:21] == "RK4MethodSolver(step="
    assert resolved_solver.step == actual_step


def test_fixed_step_falls_back_to_duration_over_100():
    # When max_spectral_radius is None, recommended = duration / 100
    actual_step, resolved_solver = resolve_default_solver(
        duration=10.0,
        step_size=None,
        solver=None,
        max_spectral_radius=None,
        require_fixed_step=True,
    )

    assert actual_step == 0.1  # 10.0 / 100
    assert resolved_solver.__repr__() == "RK4MethodSolver(step=0.1)"


def test_adaptive_solver_and_step_size_are_independent():
    solver = mini_ode.RKF45MethodSolver(
        rtol=1e-8, atol=1e-8, safety_factor=0.9, min_step=1e-10
    )

    actual_step, resolved_solver = resolve_default_solver(
        duration=1.0,
        step_size=0.05,  # Input-update interval
        solver=solver,
        max_spectral_radius=1.0,
        require_fixed_step=False,  # Adaptive mode
    )

    assert actual_step == 0.05
    assert resolved_solver is solver


def test_warns_when_max_spectral_radius_unknown():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        resolve_default_solver(
            duration=1.0,
            step_size=None,
            solver=None,
            max_spectral_radius=None,  # No spectral radius information
            require_fixed_step=True,
        )

        assert any("max_spectral_radius is unknown" in str(warn.message) for warn in w)


def test_warns_when_step_exceeds_stability_limit():
    solver = mini_ode.RK4MethodSolver(step=10.0)  # Very big step

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        resolve_default_solver(
            duration=1.0,
            step_size=None,
            solver=solver,
            max_spectral_radius=1.0,  # stability_radius ~= 2.8, so max_stable_step ~= 2.8
            require_fixed_step=True,
        )

        assert any("exceeds the stability limit" in str(warn.message) for warn in w)


def test_warns_when_step_is_unnecessarily_small():
    # For max_spectral_radius=1.0, recommended ~= 1.4
    # step_size=0.01 is much smaller than recommended/10
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        resolve_default_solver(
            duration=10.0,
            step_size=0.01,
            solver=None,
            max_spectral_radius=1.0,
            require_fixed_step=True,
        )

        assert any("significantly smaller" in str(warn.message) for warn in w)


def test_discrete_mode_snaps_to_nice_values():
    # For max_spectral_radius=1.0, recommended ~= 1.4
    # Should snap to 1.0 (largest nice value <= 1.4)
    actual_step, _ = resolve_default_solver(
        duration=10.0,
        step_size=None,
        solver=None,
        max_spectral_radius=1.0,
        require_fixed_step=True,
        discrete=True,
    )

    # Check if the step is "nice" (1, 2, 2.5, 5 x 10^n)
    nice_values = {0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0}
    assert actual_step in nice_values


def test_fixed_step_solver_without_step_attribute_raises():
    class BadSolver:
        pass

    with pytest.raises(ValueError, match="must expose a 'step' attribute"):
        resolve_default_solver(
            duration=1.0,
            step_size=None,
            solver=BadSolver(),
            max_spectral_radius=1.0,
            require_fixed_step=True,
        )


def test_fixed_step_solver_with_none_step_raises():
    class BadSolver:
        step = None

    with pytest.raises(ValueError, match="must expose a numeric 'step' attribute"):
        resolve_default_solver(
            duration=1.0,
            step_size=None,
            solver=BadSolver(),
            max_spectral_radius=1.0,
            require_fixed_step=True,
        )
