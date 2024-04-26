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

"""Utility functions for working with pytrees. See the JAX documentation page on pytrees for more
https://jax.readthedocs.io/en/latest/pytrees.html.
"""

from typing import Iterable

import numpy as np
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_map
from jax.experimental.maps import xmap

from qiskit import QiskitError


def tree_concatenate(trees: Iterable["PyTree"], axis: int = 0) -> "PyTree":
    """Given an Iterable of PyTrees with the same tree definition and whose leaves are all arrays,
    return a single PyTree whose leaves are the concatenated arrays of the inputs.

    For the concatenation to be possible, this function necessarily requires that ``leaf.ndim >= 1``
    (i.e. the leaves are not scalars), and ``leaf.shape[1:]`` is the same for each leaf across all
    trees.

    Args:
        trees: Iterable of the trees to concatenate.
        axis: Concatenation axis passed directly to ``jax.numpy.concatenate``.

    Returns:
        PyTree: The concatenated PyTree.

    Raises:
        QiskitError: If the tree definitions don't agree. If the concatenation fails due to the
        assumptions on dimension or shape for the leaves not being satisfied, an error will be
        raised directly by JAX.
    """

    leaves0, tree_def = tree_flatten(trees[0])

    leaves_list = [leaves0]
    for tree in trees[1:]:
        next_leaves, next_tree_def = tree_flatten(tree)

        if next_tree_def != tree_def:
            raise QiskitError("All trees passed to tree_stack must have the same tree definition.")

        leaves_list.append(next_leaves)

    concatenated_leaves = [jnp.concatenate(x, axis=axis) for x in zip(*leaves_list)]

    return tree_def.unflatten(concatenated_leaves)


def tree_product(trees: Iterable["PyTree"]) -> "PyTree":
    """Take the "Cartesian product" of an iterable of PyTrees along the leading axis of their
    leaves.

    The simplest usage of this function is when the trees are simple individual arrays. As an
    example, given ``a = jnp.array([1., 2., 3.])`` and ``b = jnp.array([-1, 0, 1])``,
    ``a_out, b_out = tree_product([a, b])``, it holds that
    ``a_out == jnp.array([1., 1., 1., 2., 2., 2., 3., 3., 3.])`` and
    ``b_out == jnp.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])``. I.e., ``zip(a_out, b_out)`` will iterate
    over the Cartesian product of the entries of ``a`` and ``b``.

    This behaviour is extended to PyTrees, with the restriction that within a PyTree, each leaf must
    have a leading axis of the same length. (The inputs can be viewed as a list of PyTrees that have
    been passed through :func:`tree_concatenate`.) Abusing notation, for a PyTree ``pt``, let
    ``pt[idx]`` denote ``tree_map(pt, lambda a: a[idx])`` (i.e. ``pt[idx]`` denotes the PyTree when
    indexing all leaves with ``idx``). Let ``a1``, ..., ``am`` denote PyTrees whose leaves are all
    arrays with dimension at least 1, and within each PyTree, all leaves have the same length.
    It holds that
    ``tree_product([a1, ..., am])[idx1, ..., idxm] = (a1[idx1], ..., am[idxm])``.

    For example, given
    ``a = (jnp.array([1., 2., 3.]), jnp.array([[4., 5.], [6., 7.], [8., 9.]]))`` and
    ``b = (jnp.array([0, 1]), jnp.array([0, 2]), jnp.array([3, 4]))`` and
    ``a_out, b_out = tree_product([a, b])``, it holds that
    ``a_out[0] == jnp.array([1., 1., 2., 2., 3., 3.])``
    ``a_out[1] == jnp.array([[4., 5.], [4., 5.], [6., 7.], [6., 7.], [8., 9.], [8., 9.]])``,
    ``b_out[0] == jnp.array([0, 1, 0, 1, 0, 1])``, and
    ``b_out[1] == jnp.array([0, 2, 0, 2, 0, 2])``.

    Args:
        trees: Iterably of PyTrees.
    Returns:
        PyTree: A list of PyTrees.
    Raises:
        QiskitError: if leaves of input do not satisfy the function requirements.
    """

    # validate that, within each tree, the leaves all have the same length
    for tree in trees:
        leaves, _ = tree_flatten(tree)
        if any(
            not isinstance(leaf, (np.ndarray, jnp.ndarray)) or leaf.ndim == 0 for leaf in leaves
        ):
            raise QiskitError("All pytree leaves must be arrays having dimension at least 1.")

        len0 = len(leaves[0])
        if any(len(leaf) != len0 for leaf in leaves[1:]):
            raise QiskitError(
                "pytree_product requires that all leaves within a given tree have the same "
                "length."
            )

    # compute the Cartesian product where first len(trees) leading dimensions index each combination
    outer_product_trees = xmap(
        lambda *args: args,
        in_axes=[{0: f"a{k}"} for k in range(len(trees))],
        out_axes=tuple([{k: f"a{k}" for k in range(len(trees))}] * len(trees)),
    )(*trees)

    # flatten first len(trees) dimensions
    num_trees = len(trees)

    def flatten_func(leaf):
        shape = leaf.shape
        return leaf.reshape((np.prod(shape[:num_trees]),) + shape[num_trees:])

    return tree_map(flatten_func, outer_product_trees)
