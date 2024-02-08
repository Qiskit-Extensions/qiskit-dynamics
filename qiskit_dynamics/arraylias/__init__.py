# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

r"""
=========================================
Models (:mod:`qiskit_dynamics.arraylias`)
=========================================

.. currentmodule:: qiskit_dynamics.arraylias

Module for Qiskit Dynamics global NumPy and SciPy aliases.
"""

from .alias import (
    DYNAMICS_NUMPY_ALIAS,
    DYNAMICS_SCIPY_ALIAS,
    DYNAMICS_NUMPY,
    DYNAMICS_SCIPY,
    ArrayLike,
    requires_array_library,
)
