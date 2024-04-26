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
# pylint: disable=invalid-name

"""
Tests for pytree utils.
"""

import unittest

import jax.numpy as jnp

from qiskit import QiskitError

from qiskit_dynamics.compute_utils.pytree_utils import tree_concatenate, tree_product


class Testtree_concatenate(unittest.TestCase):
    """Test tree_concatenate."""

    def test_arrays(self):
        """Test on raw arrays."""
        out = tree_concatenate([jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0])])
        self.assertTrue(all(out == jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])))

    def test_pytree_different_shapes(self):
        """Test on pytrees whose entries have different shapes."""
        tree0 = (jnp.array([1.0, 2.0]), [jnp.array([[3.0, 4.0], [5.0, 6.0]])])
        tree1 = (jnp.array([7.0, 8.0, 9.0]), [jnp.array([[10.0, 11.0]])])

        out = tree_concatenate([tree0, tree1])
        self.assertTrue(len(out) == 2)
        self.assertTrue(all(out[0] == jnp.array([1.0, 2.0, 7.0, 8.0, 9.0])))
        self.assertTrue(
            all((out[1][0] == jnp.array([[3.0, 4.0], [5.0, 6.0], [10.0, 11.0]])).flatten())
        )

    def test_tree_def_error(self):
        """Test that inconsistent tree defs results in a raised error."""
        tree0 = (1, 2)
        tree1 = (1,)

        with self.assertRaisesRegex(QiskitError, "same tree def"):
            tree_concatenate([tree0, tree1])


class Testtree_product(unittest.TestCase):
    """Test tree_product."""

    def test_arrays(self):
        """Test on raw arrays."""

        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([-1, 0, 1])
        out = tree_product([a, b])
        self.assertTrue(all(out[0] == jnp.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])))
        self.assertTrue(all(out[1] == jnp.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])))

    def test_3_arrays(self):
        """Test on raw arrays."""

        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([-1, 0])
        c = jnp.array([[0.0, 1.0]])
        out = tree_product([a, b, c])
        self.assertTrue(all(out[0] == jnp.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])))
        self.assertTrue(all(out[1] == jnp.array([-1, 0, -1, 0, -1, 0])))
        self.assertTrue(
            all(
                (
                    out[2]
                    == jnp.array(
                        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
                    )
                ).flatten()
            )
        )

    def test_1_input(self):
        """Test edge case of only one input."""

        a = jnp.array([1.0, 2.0, 3.0])
        out = tree_product([a])
        self.assertTrue(all(out[0] == a))

    def test_simple_pytree(self):
        """Test a case of two simple pytrees."""

        a = (jnp.array([1.0, 2.0, 3.0]), jnp.array([[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]]))
        b = (jnp.array([0, 1]), jnp.array([0, 2]), jnp.array([3, 4]))
        a_out, b_out = tree_product([a, b])
        self.assertTrue(all(a_out[0] == jnp.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])))
        self.assertTrue(
            all(
                (
                    a_out[1]
                    == jnp.array(
                        [[4.0, 5.0], [4.0, 5.0], [6.0, 7.0], [6.0, 7.0], [8.0, 9.0], [8.0, 9.0]]
                    )
                ).flatten()
            )
        )
        self.assertTrue(all(b_out[0] == jnp.array([0, 1, 0, 1, 0, 1])))
        self.assertTrue(all(b_out[1] == jnp.array([0, 2, 0, 2, 0, 2])))

    def test_scalar_error(self):
        """Test an error is raised if a scalar is leaf is present."""

        with self.assertRaisesRegex(QiskitError, "dimension at least 1"):
            tree_product([jnp.array(1)])

    def test_length_error(self):
        """Test an error is raised if two leaves in the same tree have different lengths."""

        with self.assertRaisesRegex(QiskitError, "all leaves within a given tree have the same"):
            tree_product([(jnp.array([1]), jnp.array([2, 3]))])
