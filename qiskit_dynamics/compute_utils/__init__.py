# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
====================================================
Compute Utils (:mod:`qiskit_dynamics.compute_utils`)
====================================================

.. currentmodule:: qiskit_dynamics.compute_utils

This submodule contains utilities to aid in running computations, and is based in JAX.
"""

from .parallel_maps import grid_map
from .pytree_utils import tree_concatenate, tree_product
