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
Tests for parallel maps. Note that we can't test actual parallelism here, we can only verify the
correctness of the output.
"""

import unittest

import numpy as np
import jax.numpy as jnp
from jax import random
from qiskit import QiskitError
from qiskit_dynamics.compute_utils.parallel_maps import (
    grid_map,
    _move_args_to_front,
    _tree_product_with_keys,
)


class Testgrid_map(unittest.TestCase):
    """Test grid_map."""

    def test_device_dim_error(self):
        """Test error is raised if device dimension is not 1."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        with self.assertRaisesRegex(QiskitError, "devices must be a 1d"):
            grid_map(jnp.sin, x, devices=np.array([[1, 2], [3, 4]]))

    def test_1d_grid(self):
        """Test correct output when run on a 1d grid."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        output = grid_map(jnp.sin, x)
        expected = jnp.sin(x)
        self.assertTrue(np.allclose(output, expected))

    def test_2d_grid(self):
        """Test correct output when run on a 2d grid."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([6.0, 7.0, 8.0, 9.0, 10.0])

        def f(x, y):
            return jnp.sin(0.1 * x + 0.2 * y)

        output = grid_map(f, x, y)
        expected = jnp.array([[f(a, b) for b in y] for a in x])
        self.assertTrue(np.allclose(output, expected))

    def test_3d_grid(self):
        """Test correct output when run on a 3d grid."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
        z = np.array([11.0, 12.0])

        def f(x, y, z):
            return jnp.sin(0.1 * x + 0.2 * y + 0.3 * z)

        output = grid_map(f, x, y, z)
        expected = jnp.array([[[f(a, b, c) for c in z] for b in y] for a in x])
        self.assertTrue(np.allclose(output, expected))

    def test_1d_grid_pytree(self):
        """Test correct function on 1d pytree grid."""
        x = (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))

        def f(a):
            return a[0] * a[1]

        output = grid_map(f, x)
        expected = jnp.array([4.0, 10.0, 18.0])
        self.assertTrue(np.allclose(output, expected))

    def test_2d_grid_pytree_output(self):
        """Test correct function on 2d pytree grid, with a pytree output."""

        x = (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
        y = {"arg0": jnp.array([[0, 1], [1, 0]]), "arg1": jnp.array([3, 4])}

        def f(a, b):
            return (a[0] * a[1], b["arg0"] * b["arg1"])

        output = grid_map(f, x, y)
        expected = (
            jnp.array([[4.0, 4.0], [10.0, 10.0], [18.0, 18.0]]),
            jnp.array([[[0, 3], [4, 0]], [[0, 3], [4, 0]], [[0, 3], [4, 0]]]),
        )
        self.assertTrue(len(output) == 2)
        self.assertTrue(isinstance(output, tuple))
        self.assertTrue(np.allclose(output[0], expected[0]))
        self.assertTrue(np.allclose(output[1], expected[1]))

    def test_arrays_of_different_shape(self):
        """Test correct mapping when input arrays are of different shape."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array(
            [[[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]
        )

        def f(x, y):
            return y @ x

        output = grid_map(f, x, y)
        expected = jnp.array(
            [[y[0] @ x[0], y[1] @ x[0], y[2] @ x[0]], [y[0] @ x[1], y[1] @ x[1], y[2] @ x[1]]]
        )
        self.assertTrue(np.allclose(output, expected))

    def test_correct_mapping_max_vmap(self):
        """Test correct mapping with a max vmap."""

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        output = grid_map(jnp.sin, x, max_vmap_size=3)
        expected = jnp.sin(x)
        self.assertTrue(np.allclose(output, expected))

    def test_key_inclusion(self):
        """Test correct handling of key generation."""

        def f(a, b, key):
            return {"a": a, "b": b, "key": key}

        key = random.PRNGKey(1234)
        output = grid_map(f, jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]), key=key)

        expected_keys = random.split(key, 4).reshape(2, 2, 2)
        self.assertTrue(np.allclose(output["a"], jnp.array([1.0, 1.0, 2.0, 2.0]).reshape(2, 2)))
        self.assertTrue(np.allclose(output["b"], jnp.array([3.0, 4.0, 3.0, 4.0]).reshape(2, 2)))
        self.assertTrue(np.allclose(output["key"], expected_keys))

    def test_key_inclusion_2_per(self):
        """Test correct handling of key generation with 2 per key."""

        def f(a, b, key):
            return {"a": a, "b": b, "key": key}

        key = random.PRNGKey(1234)
        output = grid_map(
            f, jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]), key=key, keys_per_grid_point=2
        )

        expected_keys = random.split(key, 8).reshape(2, 2, 2, 2)
        self.assertTrue(
            np.allclose(
                output["a"], jnp.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]).reshape(2, 2, 2)
            )
        )
        self.assertTrue(
            np.allclose(
                output["b"], jnp.array([3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0]).reshape(2, 2, 2)
            )
        )
        self.assertTrue(np.allclose(output["key"], expected_keys))


class Testgrid_map_nonjax_args(unittest.TestCase):
    """Test grid_map with nonjax arguments."""

    def test_non_int_argnum(self):
        """Validate error is raised for non integer argnum."""

        with self.assertRaisesRegex(QiskitError, "must be int"):
            grid_map(None, None, nonjax_argnums=["not_an_int"])

    def test_3_args_2_nonjax_args(self):
        """Test case with 3 args and 2 nonjax args."""

        def f(a, b, c):
            if b:
                if c == "string":
                    return a
                else:
                    return a**2
            else:
                if c == "string":
                    return a**3
                else:
                    return a**4

        a_list = jnp.array([2.0, 3.0, 4.0])
        b_list = [True, False]
        c_list = ["string", "notstring"]

        expected = np.zeros((len(a_list), len(b_list), len(c_list)))
        for a_idx, a in enumerate(a_list):
            for b_idx, b in enumerate(b_list):
                for c_idx, c in enumerate(c_list):
                    expected[a_idx, b_idx, c_idx] = f(a, b, c)

        self.assertTrue(
            np.allclose(expected, grid_map(f, a_list, b_list, c_list, nonjax_argnums=[1, 2]))
        )

    def test_4_args_2_nonjax_args(self):
        """Test case with 4 args and 2 nonjax args."""

        def f(a, b, c, d):
            if b:
                if c == "string":
                    return a + 1j * d
                else:
                    return a**2 + 2j * d
            else:
                if c == "string":
                    return a**3 + 3j * d
                else:
                    return a**4 + 4j * d

        a_list = jnp.array([2.0, 3.0, 4.0])
        b_list = [True, False]
        c_list = ["string", "notstring"]
        d_list = jnp.array([5.0, 6.0, 7.0, 8.0])

        expected = np.zeros((len(a_list), len(b_list), len(c_list), len(d_list)), dtype=complex)
        for a_idx, a in enumerate(a_list):
            for b_idx, b in enumerate(b_list):
                for c_idx, c in enumerate(c_list):
                    for d_idx, d in enumerate(d_list):
                        expected[a_idx, b_idx, c_idx, d_idx] = f(a, b, c, d)

        self.assertTrue(
            np.allclose(
                expected, grid_map(f, a_list, b_list, c_list, d_list, nonjax_argnames=["b", "c"])
            )
        )

    def test_4_args_2_nonjax_args_non_consecutive(self):
        """Test case with 4 args and 2 non-consecutive nonjax args."""

        def f(a, b, d, c):
            if b:
                if c == "string":
                    return a + 1j * d
                else:
                    return a**2 + 2j * d
            else:
                if c == "string":
                    return a**3 + 3j * d
                else:
                    return a**4 + 4j * d

        a_list = jnp.array([2.0, 3.0, 4.0])
        b_list = [True, False]
        c_list = ["string", "notstring"]
        d_list = jnp.array([5.0, 6.0, 7.0, 8.0])

        expected = np.zeros((len(a_list), len(b_list), len(d_list), len(c_list)), dtype=complex)
        for a_idx, a in enumerate(a_list):
            for b_idx, b in enumerate(b_list):
                for d_idx, d in enumerate(d_list):
                    for c_idx, c in enumerate(c_list):
                        expected[a_idx, b_idx, d_idx, c_idx] = f(a, b, d, c)

        self.assertTrue(
            np.allclose(
                expected, grid_map(f, a_list, b_list, d_list, c_list, nonjax_argnames=["b", "c"])
            )
        )

    def test_key_inclusion(self):
        """Test correct handling of key generation with nonjax argnums."""

        def f(a, b, key):
            return {"a": a, "b": b, "key": key}

        key = random.PRNGKey(1234)
        output = grid_map(
            f, jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0]), key=key, nonjax_argnums=[1]
        )

        nonjax_keys = random.split(key, 2)

        expected_keys = jnp.array(
            [random.split(nonjax_keys[0], 3), random.split(nonjax_keys[1], 3)]
        ).transpose((1, 0, 2))

        self.assertTrue(
            np.allclose(output["a"], jnp.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).reshape(3, 2))
        )
        self.assertTrue(
            np.allclose(output["b"], jnp.array([4.0, 5.0, 4.0, 5.0, 4.0, 5.0]).reshape(3, 2))
        )
        self.assertTrue(np.allclose(output["key"], expected_keys))

    def test_key_inclusion_2_per(self):
        """Test correct handling of key generation with nonjax argnums and 2 keys per input."""

        def f(a, b, key):
            return {"a": a, "b": b, "key": key}

        key = random.PRNGKey(1234)
        output = grid_map(
            f,
            jnp.array([1.0, 2.0, 3.0]),
            jnp.array([4.0, 5.0]),
            key=key,
            keys_per_grid_point=2,
            nonjax_argnums=[1],
        )

        nonjax_keys = random.split(key, 2)

        expected_keys = (
            jnp.array([random.split(nonjax_keys[0], 6), random.split(nonjax_keys[1], 6)])
            .reshape(2, 3, 2, 2)
            .transpose((1, 0, 2, 3))
        )

        self.assertTrue(
            np.allclose(
                output["a"],
                jnp.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]).reshape(
                    3, 2, 2
                ),
            )
        )
        self.assertTrue(
            np.allclose(
                output["b"],
                jnp.array([4.0, 4.0, 5.0, 5.0, 4.0, 4.0, 5.0, 5.0, 4.0, 4.0, 5.0, 5.0]).reshape(
                    3, 2, 2
                ),
            )
        )
        self.assertTrue(np.allclose(output["key"], expected_keys))


class Test_move_args_to_front(unittest.TestCase):
    """Tests for helper function _move_args_to_front."""

    def test_array_building_func_case1(self):
        """Test a function that compiles scalar inputs into a 1d array."""

        def f(a, b, c, d, e):
            return np.array([a, b, c, d, e])

        g = _move_args_to_front(f, argnums=[1, 4])

        #       b  e  a  c  d
        out = g(1, 2, 3, 4, 5)
        expected = np.array([3, 1, 4, 5, 2])

        self.assertTrue(np.allclose(out, expected))

    def test_array_building_func_case2(self):
        """Test a function that compiles scalar inputs into a 1d array."""

        def f(a, b, c, d, e):
            return np.array([a, b, c, d, e])

        g = _move_args_to_front(f, argnums=[1, 3])

        #       b  d  a  c  e
        out = g(1, 2, 3, 4, 5)
        expected = np.array([3, 1, 4, 2, 5])

        self.assertTrue(np.allclose(out, expected))


class Test_tree_product_with_keys(unittest.TestCase):
    """Test cases for _tree_product_with_keys."""

    def test_invalid_key(self):
        """Test key of incorrect type."""

        with self.assertRaisesRegex(QiskitError, "Invalid format"):
            _tree_product_with_keys((jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])), key="key")

    def test_case1(self):
        """Simple test case."""
        key = random.PRNGKey(1234)

        output = _tree_product_with_keys(
            (jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])),
            key=key,
        )

        expected_keys = random.split(key, 4)
        self.assertTrue(isinstance(output, tuple) and len(output) == 3)
        self.assertTrue(np.allclose(output[0], np.array([1.0, 1.0, 2.0, 2.0])))
        self.assertTrue(np.allclose(output[1], np.array([3.0, 4.0, 3.0, 4.0])))
        self.assertTrue(np.allclose(output[2], expected_keys))

    def test_case_2_per(self):
        """Test case with with 2 keys per input."""
        key = random.PRNGKey(1234)

        output = _tree_product_with_keys(
            (jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])), key=key, keys_per_grid_point=2
        )

        expected_keys = random.split(key, 8)
        self.assertTrue(isinstance(output, tuple) and len(output) == 3)
        self.assertTrue(np.allclose(output[0], np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])))
        self.assertTrue(np.allclose(output[1], np.array([3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0])))
        self.assertTrue(np.allclose(output[2], expected_keys))
