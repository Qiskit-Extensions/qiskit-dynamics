# -*- coding: utf-8 -*-

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

"""
==========================================
Qiskit Dynamics (:mod:`qiskit_dynamics`)
==========================================

.. currentmodule:: qiskit_dynamics

Qiskit extension module for simulating quantum dynamics.
"""
from .version import __version__

from .arraylias.alias import (
    DYNAMICS_NUMPY_ALIAS,
    DYNAMICS_SCIPY_ALIAS,
    DYNAMICS_NUMPY,
    DYNAMICS_SCIPY,
    ArrayLike,
)

from .models.rotating_frame import RotatingFrame

from .signals.signals import Signal, DiscreteSignal

from .solvers.solver_functions import solve_ode, solve_lmde
from .solvers.solver_classes import Solver
from .solvers.perturbative_solvers.dyson_solver import DysonSolver
from .solvers.perturbative_solvers.magnus_solver import MagnusSolver
from .backend.dynamics_backend import DynamicsBackend

from . import models
from . import signals
from . import pulse
from . import backend
from . import compute_utils
