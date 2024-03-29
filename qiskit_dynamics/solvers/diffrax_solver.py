# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
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
Wrapper for diffrax solvers
"""

from typing import Callable, Optional
from scipy.integrate._ivp.ivp import OdeResult
from qiskit import QiskitError

from qiskit_dynamics.arraylias import ArrayLike, requires_array_library

try:
    import jax.numpy as jnp
except ImportError:
    pass


@requires_array_library("jax")
def diffrax_solver(
    rhs: Callable,
    t_span: ArrayLike,
    y0: ArrayLike,
    method: "AbstractSolver",
    t_eval: Optional[ArrayLike] = None,
    **kwargs,
):
    """Routine for calling ``diffrax.diffeqsolve``

    Args:
        rhs: Callable of the form :math:`f(t, y)`.
        t_span: Interval to solve over.
        y0: Initial state.
        method: Which diffeq solving method to use.
        t_eval: Optional list of time points at which to return the solution.
        **kwargs: Optional arguments to be passed to ``diffeqsolve``.

    Returns:
        OdeResult: Results object.

    Raises:
        QiskitError: Passing both `SaveAt` argument and `t_eval` argument.
    """

    from diffrax import ODETerm, SaveAt
    from diffrax import diffeqsolve

    # convert rhs and y0 to real
    rhs = real_rhs(rhs)
    y0 = c2r(y0)

    term = ODETerm(lambda t, y, _: rhs(t.real, y))

    if "saveat" in kwargs and t_eval is not None:
        raise QiskitError(
            """Only one of t_eval or saveat can be passed when using
            a diffrax solver, but both were specified."""
        )

    if t_eval is not None:
        kwargs["saveat"] = SaveAt(ts=t_eval)

    results = diffeqsolve(
        term,
        solver=method,
        t0=t_span[0],
        t1=t_span[-1],
        dt0=None,
        y0=jnp.array(y0, dtype=float),
        **kwargs,
    )

    sol_dict = vars(results)
    ys = sol_dict.pop("ys")
    ts = sol_dict.pop("ts")

    ys = jnp.swapaxes(r2c(jnp.swapaxes(ys, 0, 1)), 0, 1)

    results_out = OdeResult(t=ts, y=jnp.array(ys, dtype=complex), **sol_dict)

    return results_out


def real_rhs(rhs):
    """Convert complex RHS to real RHS function"""

    def _real_rhs(t, y):
        return c2r(rhs(t, r2c(y)))

    return _real_rhs


def c2r(arr):
    """Convert complex array to a real array"""
    return jnp.concatenate([jnp.real(arr), jnp.imag(arr)])


def r2c(arr):
    """Convert a real array to a complex array"""
    size = arr.shape[0] // 2
    return arr[:size] + 1j * arr[size:]
