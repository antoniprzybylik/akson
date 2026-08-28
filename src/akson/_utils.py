import mini_ode
import warnings
import math
from typing import Optional, Tuple


def _round_to_nice(value: float) -> float:
    """Snap a step size down to a human-friendly value.

    Returns the largest value in {1, 2, 2.5, 5} × 10^n that does not
    exceed the input.
    """
    if value <= 0:
        return value

    exponent = math.floor(math.log10(value))
    mantissa = value / (10.0**exponent)
    candidates = [1.0, 2.0, 2.5, 5.0]

    for candidate in reversed(candidates):
        if mantissa >= candidate:
            return candidate * (10.0**exponent)

    return 1.0 * (10.0**exponent)


def resolve_default_solver(
    duration: float,
    step_size: Optional[float],
    solver: Optional[object],
    max_spectral_radius: Optional[float],
    require_fixed_step: bool = False,
    discrete: bool = False,
) -> Tuple[Optional[float], object]:
    """Resolve solver and step size.

    The function operates in four phases:
    1. Sanity checks on arguments.
    2. Determine the solver instance and step size following the precedence rules.
    3. Create a default solver is none was suplied
    4. Emit diagnostic warnings about stability, speed or missing spectral radius.

    Precedence rules:
    Case 1: require_fixed_step=True:
        The integration step is authoritative and comes from:
            1. Provided solver's `step` attribute
            2. Provided `step_size` argument  (only when no solver given)
            3. Stability-based recommendation (only when neither given)
            4. Duration / 100                 (last resort)
        If both `solver` and `step_size` are provided, the solver wins
        and `step_size` is ignored (UserWarning).

    Case 2: require_fixed_step=False:
        The solver manages its own internal step. `step_size` is only
        the input-update interval:
            1. Provided `step_size` argument
            2. Stability-based recommendation
            3. Duration / 100
        `solver` and `step_size` are independent and do not conflict.

    Stability based recommendation:
    Whenever `max_spectral_radius` is known (always for LTI systems -
    computed exactly from A's eigenvalues; optionally supplied by the
    caller for nonlinear systems, see StateSpaceDynamics.__init__) AND the
    resolved solver reports a finite `stability_radius`:

        recommended = stability_radius / max_spectral_radius / 2

    Otherwise, the recommendation is purely resolution-driven:

        recommended = duration / 100

    This fallback fires whenever either piece of information is missing:
      - `max_spectral_radius` is unknown (the common case for a nonlinear
        system that hasn't been given one), regardless of the solver, OR
      - the solver's `stability_radius` is infinite (e.g. an implicit,
        A-stable method that has no linear-stability step restriction at
        all), regardless of whether `max_spectral_radius` is known.

    If discrete=True, the recommended step is snapped to a human-friendly value.
    """
    # Sanity checks on arguments
    if duration <= 0:
        raise ValueError("Duration must be positive")
    if step_size is not None and step_size <= 0:
        raise ValueError("Step size must be positive")
    if max_spectral_radius is not None and max_spectral_radius <= 0:
        raise ValueError("Spectral radius must be positive")

    # Determine the solver if it was already supplied or step size is not needed
    if solver is not None:
        # Caller supplied a solver instance, validation is performed
        if require_fixed_step and not hasattr(solver, "step"):
            raise ValueError(
                "Fixed-step solver must expose a 'step' attribute. "
                f"Got {type(solver).__name__} which lacks it."
            )
        resolved_solver = solver
    else:
        # No solver supplied.  For adaptive mode we can create the final
        # instance immediately; for fixed-step we defer instantiation
        # until we know the step.
        if require_fixed_step:
            resolved_solver = None
        else:
            resolved_solver = mini_ode.RKF45MethodSolver(
                rtol=1e-6, atol=1e-9, min_step=1e-12, safety_factor=0.9
            )

    # Get the stability radius
    if solver is not None:
        stability_radius = getattr(solver, "stability_radius", float("inf"))
    elif require_fixed_step:
        # Probe the default RK4 type for its stability property
        _probe = mini_ode.RK4MethodSolver(step=1.0)
        stability_radius = getattr(_probe, "stability_radius", float("inf"))
    else:
        stability_radius = getattr(resolved_solver, "stability_radius", float("inf"))

    has_stability_info = (
        max_spectral_radius is not None
        and not math.isinf(stability_radius)
        and max_spectral_radius > 0
    )

    # Compute the recommended step
    if has_stability_info:
        recommended_step = stability_radius / max_spectral_radius / 2.0
    else:
        recommended_step = duration / 100.0

    if discrete:
        recommended_step = _round_to_nice(recommended_step)

    # Resolve the actual step
    if require_fixed_step:
        # Precedence: solver.step > step_size > recommended
        if solver is not None:
            if step_size is not None:
                warnings.warn(
                    "Both 'solver' and 'step_size' were provided for a fixed-step "
                    "integration. The solver's own 'step' attribute takes precedence; "
                    "'step_size' is ignored.",
                    UserWarning,
                )
            actual_step = getattr(solver, "step", None)
            if actual_step is None:
                raise ValueError(
                    "Fixed-step solver must expose a numeric 'step' attribute."
                )
        elif step_size is not None:
            actual_step = step_size
        else:
            actual_step = recommended_step
    else:
        # Adaptive: step_size is the input-update interval
        actual_step = step_size if step_size is not None else recommended_step

    if actual_step is not None and actual_step <= 0:
        raise ValueError("Resolved step size must be positive")

    # Default fixed step solver
    if solver is None and require_fixed_step:
        resolved_solver = mini_ode.RK4MethodSolver(step=actual_step)

    # Validate fixed step solvers for step
    if require_fixed_step and getattr(resolved_solver, "step", None) is None:
        raise ValueError("Fixed-step solver must expose a numeric 'step' attribute.")

    # Emit diagnostic warnings about stability, speed or missing spectral radius.
    final_stability = getattr(resolved_solver, "stability_radius", float("inf"))
    final_has_stability = (
        max_spectral_radius is not None
        and not math.isinf(final_stability)
        and max_spectral_radius > 0
    )

    if max_spectral_radius is None:
        warnings.warn(
            "max_spectral_radius is unknown for this (nonlinear) system. "
            "Stability is not guaranteed - choose solver parameters carefully, "
            "or supply max_spectral_radius to StateSpaceDynamics for an "
            "informed, stability-based step-size choice and warnings.",
            RuntimeWarning,
        )
    elif actual_step is not None and final_has_stability:
        max_stable_step = final_stability / max_spectral_radius
        if require_fixed_step and actual_step > max_stable_step:
            warnings.warn(
                f"Step size {actual_step:.4e} exceeds the stability limit "
                f"{max_stable_step:.4e} for this solver and system. "
                f"Results may be unstable.",
                RuntimeWarning,
            )
        elif require_fixed_step and actual_step < recommended_step / 10.0:
            warnings.warn(
                f"Step size {actual_step:.4e} is significantly smaller than "
                f"the recommended {recommended_step:.4e} (stability limit "
                f"{max_stable_step:.4e}). Simulation may be unnecessarily slow.",
                RuntimeWarning,
            )
        elif (
            not require_fixed_step
            and step_size is not None
            and actual_step > max_stable_step
        ):
            # For adaptive solvers, step_size is the update interval.
            # The solver subdivides internally, but an extremely coarse
            # interval may miss fast dynamics or produce poor output resolution.
            warnings.warn(
                f"Input-update interval {actual_step:.4e} exceeds the stability "
                f"scale {max_stable_step:.4e}. The adaptive solver will manage "
                f"internal steps, but output resolution may be coarse.",
                RuntimeWarning,
            )

    return actual_step, resolved_solver
