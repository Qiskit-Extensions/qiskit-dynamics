# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name

"""
Custom fixed step solvers.
"""

from typing import Callable, Optional, Tuple
from warnings import warn
import numpy as np
from scipy.integrate._ivp.ivp import OdeResult
from scipy.linalg import expm

from qiskit import QiskitError

from qiskit_dynamics import DYNAMICS_NUMPY as unp
from qiskit_dynamics.arraylias import ArrayLike, requires_array_library

try:
    import jax
    from jax import vmap
    import jax.numpy as jnp
    from jax.lax import scan, cond, associative_scan
    from jax.scipy.linalg import expm as jexpm
except ImportError:
    pass

from .solver_utils import merge_t_args, trim_t_results
from .lanczos import lanczos_expm
from .lanczos import jax_lanczos_expm


def RK4_solver(
    rhs: Callable,
    t_span: ArrayLike,
    y0: ArrayLike,
    max_dt: float,
    t_eval: Optional[ArrayLike] = None,
):
    """Fixed step RK4 solver.

    Args:
        rhs: Callable, either a generator rhs
        t_span: Interval to solve over.
        y0: Initial state.
        max_dt: Maximum step size.
        t_eval: Optional list of time points at which to return the solution.

    Returns:
        OdeResult: Results object.
    """
    div6 = 1.0 / 6

    def take_step(rhs_func, t, y, h):
        h2 = 0.5 * h
        t_plus_h2 = t + h2

        k1 = rhs_func(t, y)
        k2 = rhs_func(t_plus_h2, y + h2 * k1)
        k3 = rhs_func(t_plus_h2, y + h2 * k2)
        k4 = rhs_func(t + h, y + h * k3)

        return y + div6 * h * (k1 + 2 * k2 + 2 * k3 + k4)

    return fixed_step_solver_template(
        take_step, rhs_func=rhs, t_span=t_span, y0=y0, max_dt=max_dt, t_eval=t_eval
    )


def scipy_expm_solver(
    generator: Callable,
    t_span: ArrayLike,
    y0: ArrayLike,
    max_dt: float,
    t_eval: Optional[ArrayLike] = None,
    magnus_order: int = 1,
):
    """Fixed-step size matrix exponential based solver implemented with
    ``scipy.linalg.expm``. Solves the specified problem by taking steps of
    size no larger than ``max_dt``.

    Args:
        generator: Generator for the LMDE.
        t_span: Interval to solve over.
        y0: Initial state.
        max_dt: Maximum step size.
        t_eval: Optional list of time points at which to return the solution.
        magnus_order: The expansion order in the Magnus method. Only orders 1, 2 and 3
            are supported.

    Returns:
        OdeResult: Results object.
    """
    take_step = get_exponential_take_step(magnus_order, expm_func=expm)

    return fixed_step_solver_template(
        take_step, rhs_func=generator, t_span=t_span, y0=y0, max_dt=max_dt, t_eval=t_eval
    )


def lanczos_diag_solver(
    generator: Callable,
    t_span: ArrayLike,
    y0: ArrayLike,
    max_dt: float,
    k_dim: int,
    t_eval: Optional[ArrayLike] = None,
):
    """Fixed-step size matrix exponential based solver implemented using
    lanczos algorithm. Solves the specified problem by taking steps of
    size no larger than ``max_dt``.

    Args:
        generator: Generator for the LMDE.
        t_span: Interval to solve over.
        y0: Initial state.
        max_dt: Maximum step size.
        k_dim: Integer which specifies the dimension of Krylov subspace used for
            lanczos iteration. Acts as an accuracy parameter. ``k_dim`` must
            always be less than or equal to dimension of generator.
        t_eval: Optional list of time points at which to return the solution.

    Returns:
        OdeResult: Results object.
    """

    def take_step(generator, t0, y, h):
        eval_time = t0 + (h / 2)
        return lanczos_expm(generator(eval_time), y, k_dim, h)

    return fixed_step_solver_template(
        take_step, rhs_func=generator, t_span=t_span, y0=y0, max_dt=max_dt, t_eval=t_eval
    )


@requires_array_library("jax")
def jax_lanczos_diag_solver(
    generator: Callable,
    t_span: ArrayLike,
    y0: ArrayLike,
    max_dt: float,
    k_dim: int,
    t_eval: Optional[ArrayLike] = None,
):
    """JAX version of lanczos_diag_solver."""

    def take_step(generator, t0, y, h):
        eval_time = t0 + (h / 2)
        return jax_lanczos_expm(generator(eval_time), y, k_dim, h)

    y0 = unp.asarray(y0, dtype=complex)

    return fixed_step_solver_template_jax(
        take_step, rhs_func=generator, t_span=t_span, y0=y0, max_dt=max_dt, t_eval=t_eval
    )


@requires_array_library("jax")
def jax_RK4_solver(
    rhs: Callable,
    t_span: ArrayLike,
    y0: ArrayLike,
    max_dt: float,
    t_eval: Optional[ArrayLike] = None,
):
    """JAX version of RK4_solver.

    Args:
        rhs: Callable, either a generator rhs
        t_span: Interval to solve over.
        y0: Initial state.
        max_dt: Maximum step size.
        t_eval: Optional list of time points at which to return the solution.

    Returns:
        OdeResult: Results object.
    """
    div6 = 1.0 / 6

    def take_step(rhs_func, t, y, h):
        h2 = 0.5 * h
        t_plus_h2 = t + h2

        k1 = rhs_func(t, y)
        k2 = rhs_func(t_plus_h2, y + h2 * k1)
        k3 = rhs_func(t_plus_h2, y + h2 * k2)
        k4 = rhs_func(t + h, y + h * k3)

        return y + div6 * h * (k1 + 2 * k2 + 2 * k3 + k4)

    return fixed_step_solver_template_jax(
        take_step, rhs_func=rhs, t_span=t_span, y0=y0, max_dt=max_dt, t_eval=t_eval
    )


@requires_array_library("jax")
def jax_RK4_parallel_solver(
    generator: Callable,
    t_span: ArrayLike,
    y0: ArrayLike,
    max_dt: float,
    t_eval: Optional[ArrayLike] = None,
):
    """Parallel version of :func:`jax_RK4_solver` specialized to LMDEs.

    Args:
        generator: Generator of the LMDE.
        t_span: Interval to solve over.
        y0: Initial state.
        max_dt: Maximum step size.
        t_eval: Optional list of time points at which to return the solution.

    Returns:
        OdeResult: Results object.
    """
    dim = y0.shape[-1]
    ident = jnp.eye(dim, dtype=complex)

    div6 = 1.0 / 6

    def take_step(generator, t, h):
        h2 = 0.5 * h
        gh2 = generator(t + h2)

        k1 = generator(t)
        k2 = gh2 @ (ident + h2 * k1)
        k3 = gh2 @ (ident + h2 * k2)
        k4 = generator(t + h) @ (ident + h * k3)

        return ident + div6 * h * (k1 + 2 * k2 + 2 * k3 + k4)

    return fixed_step_lmde_solver_parallel_template_jax(
        take_step, generator=generator, t_span=t_span, y0=y0, max_dt=max_dt, t_eval=t_eval
    )


@requires_array_library("jax")
def jax_expm_solver(
    generator: Callable,
    t_span: ArrayLike,
    y0: ArrayLike,
    max_dt: float,
    t_eval: Optional[ArrayLike] = None,
    magnus_order: int = 1,
):
    """Fixed-step size matrix exponential based solver implemented with ``jax``.
    Solves the specified problem by taking steps of size no larger than ``max_dt``.

    Args:
        generator: Generator for the LMDE.
        t_span: Interval to solve over.
        y0: Initial state.
        max_dt: Maximum step size.
        t_eval: Optional list of time points at which to return the solution.
        magnus_order: The expansion order in the Magnus method. Only orders 1, 2, and 3
            are supported.

    Returns:
        OdeResult: Results object.
    """
    take_step = get_exponential_take_step(magnus_order, expm_func=jexpm)

    return fixed_step_solver_template_jax(
        take_step, rhs_func=generator, t_span=t_span, y0=y0, max_dt=max_dt, t_eval=t_eval
    )


@requires_array_library("jax")
def jax_expm_parallel_solver(
    generator: Callable,
    t_span: ArrayLike,
    y0: ArrayLike,
    max_dt: float,
    t_eval: Optional[ArrayLike] = None,
    magnus_order: int = 1,
):
    """Parallel version of :func:`jax_expm_solver` implemented with JAX parallel operations.

    Args:
        generator: Generator for the LMDE.
        t_span: Interval to solve over.
        y0: Initial state.
        max_dt: Maximum step size.
        t_eval: Optional list of time points at which to return the solution.
        magnus_order: The expansion order in the Magnus method. Only orders 1, 2, and 3
            are supported.

    Returns:
        OdeResult: Results object.
    """
    take_step = get_exponential_take_step(magnus_order, expm_func=jexpm, just_propagator=True)

    return fixed_step_lmde_solver_parallel_template_jax(
        take_step, generator=generator, t_span=t_span, y0=y0, max_dt=max_dt, t_eval=t_eval
    )


def matrix_commutator(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    """Compute the commutator of two matrices.

    Args:
        m1: First argument to the commutator.
        m2: Second argument to the commutator.

    Returns:
        Matrix commutator of ``m1`` and ``m2``.
    """
    return m1 @ m2 - m2 @ m1


def get_exponential_take_step(
    magnus_order: int, expm_func: Callable, just_propagator: bool = False
):
    """Return a function implementing the infinitessimal magnus solver at 1st, 2nd, and 3rd
    Magnus orders, specified by the user. See also the documentation of :func:`scipy_expm_solver`
    for details.

    Args:
        magnus_order: The expansion order in the Magnus method. Only accepts values in
            ``[1, 2, 3]``.
        expm_func: Method of matrix exponentian.
        just_propagator: Whether or not to return function that only computes propagator.
            If False, returns a function with signature f(generator, t0, y, h), and if True, returns
            a function with signature f(generator, t0, h).

    Returns:
        take_step: Infinitessimal exponential Magnus solver.

    Raises:
        QiskitError: If ``magnus_order`` not in ``[1, 2, 3]``.
    """
    # if clause based on magnus order
    if magnus_order == 1:

        def propagator(generator, t0, h):
            return expm_func(generator(t0 + (h / 2)) * h)

    elif magnus_order == 2:
        # second-order step size constants
        c1 = 0.5 - np.sqrt(3) / 6
        c2 = 0.5 + np.sqrt(3) / 6
        p2 = np.sqrt(3) / 12

        def propagator(generator, t0, h):
            # midpoint generator calls
            g1 = generator(t0 + c1 * h)
            g2 = generator(t0 + c2 * h)

            # Magnus terms
            terms = h * (g1 + g2) / 2 + p2 * (h**2) * matrix_commutator(g2, g1)

            # solution
            return expm_func(terms)

    elif magnus_order == 3:
        # third-order step size constants
        d1 = 0.5 - np.sqrt(15) / 10
        d2 = 0.5
        d3 = 0.5 + np.sqrt(15) / 10
        c0 = np.sqrt(15) / 3
        c1 = 10.0 / 3

        def propagator(generator, t0, h):
            # midpoint generator calls
            g1 = generator(t0 + d1 * h)
            g2 = generator(t0 + d2 * h)
            g3 = generator(t0 + d3 * h)

            # linear combinations of generators
            a1 = h * g2
            a2 = c0 * h * (g3 - g1)
            a3 = c1 * h * (g3 - 2 * g2 + g1)

            # intermediate commutators
            comm1 = matrix_commutator(a1, a2)
            comm2 = matrix_commutator(2 * a3 + comm1, a1) / 60

            # Magnus terms
            terms = a1 + (a3 / 12) + matrix_commutator(-20 * a1 - a3 + comm1, a2 + comm2) / 240

            # solution
            return expm_func(terms)

    else:
        raise QiskitError("Only magnus_order 1, 2, and 3 are supported.")

    if just_propagator:
        return propagator

    def take_step(generator, t0, y, h):
        return propagator(generator, t0, h) @ y

    return take_step


def fixed_step_solver_template(
    take_step: Callable,
    rhs_func: Callable,
    t_span: ArrayLike,
    y0: ArrayLike,
    max_dt: float,
    t_eval: Optional[ArrayLike] = None,
):
    """Helper function for implementing fixed-step solvers supporting both
    ``t_span`` and ``max_dt`` arguments. ``take_step`` is assumed to be a
    function implementing a single step of size h of a fixed-step method.
    The signature of ``take_step`` is assumed to be:
        - rhs_func: Either a generator :math:`G(t)` or RHS function :math:`f(t,y)`.
        - t0: The current time.
        - y0: The current state.
        - h: The size of the step to take.

    It returns:
        - y: The state of the DE at time t0 + h.

    ``take_step`` is used to integrate the DE specified by ``rhs_func``
    through all points in ``t_eval``, taking steps no larger than ``max_dt``.
    Each interval in ``t_eval`` is divided into the least number of sub-intervals
    of equal length so that the sub-intervals are smaller than ``max_dt``.

    Args:
        take_step: Callable for fixed step integration.
        rhs_func: Callable, either a generator or rhs function.
        t_span: Interval to solve over.
        y0: Initial state.
        max_dt: Maximum step size.
        t_eval: Optional list of time points at which to return the solution.

    Returns:
        OdeResult: Results object.
    """

    y0 = unp.asarray(y0)

    t_list, h_list, n_steps_list = get_fixed_step_sizes(t_span, t_eval, max_dt)

    ys = [y0]
    for current_t, h, n_steps in zip(t_list, h_list, n_steps_list):
        y = ys[-1]
        inner_t = current_t
        for _ in range(n_steps):
            y = take_step(rhs_func, inner_t, y, h)
            inner_t = inner_t + h
        ys.append(y)
    ys = unp.asarray(ys)

    results = OdeResult(t=t_list, y=ys)

    return trim_t_results(results, t_eval)


def fixed_step_solver_template_jax(
    take_step: Callable,
    rhs_func: Callable,
    t_span: ArrayLike,
    y0: ArrayLike,
    max_dt: float,
    t_eval: Optional[ArrayLike] = None,
):
    """This function is the jax control-flow version of
    :meth:`fixed_step_solver_template`. See the documentation of :meth:`fixed_step_solver_template`
    for details.

    Args:
        take_step: Callable for fixed step integration.
        rhs_func: Callable, either a generator or rhs function.
        t_span: Interval to solve over.
        y0: Initial state.
        max_dt: Maximum step size.
        t_eval: Optional list of time points at which to return the solution.

    Returns:
        OdeResult: Results object.
    """

    y0 = jnp.array(y0)

    t_list, h_list, n_steps_list = get_fixed_step_sizes(t_span, t_eval, max_dt)

    # if jax, need bound on number of iterations in each interval
    max_steps = n_steps_list.max()

    def identity(y):
        return y

    # interval integrator set up for jax.lax.scan
    def scan_interval_integrate(carry, x):
        current_t, h, n_steps = x
        current_y = carry

        def scan_take_step(carry, step):
            t, y = carry
            y = cond(step < n_steps, lambda y: take_step(rhs_func, t, y, h), identity, y)
            t = t + h
            return (t, y), None

        next_y = scan(scan_take_step, (current_t, current_y), jnp.arange(max_steps))[0][1]

        return next_y, next_y

    ys = scan(
        scan_interval_integrate,
        init=y0,
        xs=(jnp.array(t_list[:-1]), jnp.array(h_list), jnp.array(n_steps_list)),
    )[1]

    ys = jnp.append(jnp.expand_dims(y0, axis=0), ys, axis=0)

    results = OdeResult(t=t_list, y=ys)

    return trim_t_results(results, t_eval)


def fixed_step_lmde_solver_parallel_template_jax(
    take_step: Callable,
    generator: Callable,
    t_span: ArrayLike,
    y0: ArrayLike,
    max_dt: float,
    t_eval: Optional[ArrayLike] = None,
):
    """Parallelized and LMDE specific version of fixed_step_solver_template_jax.

    Assuming the structure of an LMDE:
    * Computes all propagators over each individual time-step in parallel using ``jax.vmap``.
    * Computes all propagators from t_span[0] to each intermediate time point in parallel
      using ``jax.lax.associative_scan``.
    * Applies results to y0 and extracts the desired time points from ``t_eval``.

    The above logic is slightly varied to save some operations is ``y0`` is square.

    The signature of ``take_step`` is assumed to be:
        - generator: A generator :math:`G(t)`.
        - t: The current time.
        - h: The size of the step to take.

    It returns:
        - y: The state of the DE at time t + h.

    Note that this differs slightly from the other template functions, in that
    ``take_step`` does not take take in ``y``, the state at time ``t``. The
    parallelization procedure described above uses the initial state being the identity
    matrix for each time step, and thus it is unnecessary to supply this to ``take_step``.

    Args:
        take_step: Fixed step integration rule.
        generator: Generator for the LMDE.
        t_span: Interval to solve over.
        y0: Initial state.
        max_dt: Maximum step size.
        t_eval: Optional list of time points at which to return the solution.

    Returns:
        OdeResult: Results object.
    """

    # warn the user that the parallel solver will be very slow if run on a cpu
    if jax.default_backend() == "cpu":
        warn(
            """JAX parallel solvers will likely run slower on CPUs than non-parallel solvers.
            To make use of their capabilities it is recommended to use a GPU.""",
            stacklevel=2,
        )

    y0 = jnp.array(y0)

    t_list, h_list, n_steps_list = get_fixed_step_sizes(t_span, t_eval, max_dt)

    # set up time information for computing propagators in parallel
    all_times = []  # all stepping points
    all_h = []  # step sizes for each point above
    t_list_locations = [0]  # ordered list of locations in all_times that are in t_list
    for t, h, n_steps in zip(t_list, h_list, n_steps_list):
        all_times = np.append(all_times, t + h * np.arange(n_steps))
        all_h = np.append(all_h, h * np.ones(n_steps))
        t_list_locations = np.append(t_list_locations, [t_list_locations[-1] + n_steps])

    # compute propagators over each time step in parallel
    step_propagators = vmap(lambda t, h: take_step(generator, t, h))(all_times, all_h)

    # multiply propagators together in parallel
    ys = None

    def reverse_mul(A, B):
        return jnp.matmul(B, A)

    if y0.ndim == 2 and y0.shape[0] == y0.shape[1]:
        # if square, append y0 as the first step propagator, scan, and extract
        intermediate_props = associative_scan(
            reverse_mul, jnp.append(jnp.array([y0]), step_propagators, axis=0), axis=0
        )
        ys = intermediate_props[t_list_locations]
    else:
        # if not square, scan propagators, extract relevant time points, multiply by y0,
        # then prepend y0
        intermediate_props = associative_scan(reverse_mul, step_propagators, axis=0)
        # intermediate_props doesn't include t0, so shift t_list_locations when extracting
        intermediate_y = intermediate_props[t_list_locations[1:] - 1] @ y0
        ys = jnp.append(jnp.array([y0]), intermediate_y, axis=0)

    results = OdeResult(t=t_list, y=ys)

    return trim_t_results(results, t_eval)


def get_fixed_step_sizes(
    t_span: ArrayLike, t_eval: ArrayLike, max_dt: float
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Merge ``t_span`` and ``t_eval``, and determine the number of time steps and
    and step sizes (no larger than ``max_dt``) required to fixed-step integrate between
    each time point.

    Args:
        t_span: Total interval of integration.
        t_eval: Time points within t_span at which the solution should be returned.
        max_dt: Max size step to take.

    Returns:
        Tuple[Array, Array, Array]: with merged time point list, list of step sizes to take
        between time points, and list of corresponding number of steps to take between time steps.
    """
    # time args are non-differentiable
    t_span = np.array(t_span)
    max_dt = np.array(max_dt)
    t_list = np.array(merge_t_args(t_span, t_eval))

    # set the number of time steps required in each interval so that
    # no steps larger than max_dt are taken
    delta_t_list = np.diff(t_list)
    n_steps_list = np.abs(delta_t_list / max_dt).astype(int)

    # correct potential rounding errors
    for idx, (delta_t, n_steps) in enumerate(zip(delta_t_list, n_steps_list)):
        if n_steps == 0:
            n_steps_list[idx] = 1
        # absolute value to handle backwards integration
        elif np.abs(delta_t / n_steps) / max_dt > 1 + 1e-15:
            n_steps_list[idx] = n_steps + 1

    # step size in each interval
    h_list = np.array(delta_t_list / n_steps_list)

    return t_list, h_list, n_steps_list
