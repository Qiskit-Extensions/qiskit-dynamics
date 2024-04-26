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
# pylint: disable=invalid-name,no-member

"""
Utilities for mapping functions over arrays in parallel.
"""

from typing import Callable, Optional, Tuple, List
from itertools import product
from functools import partial
import inspect

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_flatten, tree_map
from jax.sharding import Mesh
from jax.experimental.maps import xmap

from qiskit import QiskitError

from .pytree_utils import tree_concatenate, tree_product


def grid_map(
    f: Callable,
    *args: Tuple["PyTree"],
    devices: Optional[np.array] = None,
    max_vmap_size: Optional[int] = None,
    nonjax_argnums: Optional[List[int]] = None,
    nonjax_argnames: Optional[List[str]] = None,
    key: Optional[jnp.ndarray] = None,
    keys_per_grid_point: int = 1,
) -> "PyTree":
    """Map a function ``f`` over all combinations of inputs specified by the positional arguments,
    utilizing a mix of device parallelization and vectorization.

    This function evaluates a function ``f`` with multiple inputs over a grid of input values. For
    example, suppose we have a function ``f`` of two inputs, the first being a scalar and the second
    being an array of shape ``(2,)``, and whose output is an array. We want to evaluate ``f(a, b)``
    for all combinations of ``a`` in ``a_vals = jnp.array([1, 2])`` and ``b`` in
    ``b_vals = jnp.array([[3, 4], [5, 6], [7, 8]])``. This can be done with ``grid_map`` as follows:

    .. code-block:: python

        out = grid_map(f, a_vals, b_vals)
        out == jnp.array([
                [f(1, [3, 4]), f(1, [5, 6]), f(1, [7, 8]])],
                [f(2, [3, 4]), f(2, [5, 6]), f(2, [7, 8]])]
        ])

    Note that the above output array ``out`` satisfies ``out[i, j] == f(a[i], b[j])``.

    More generally, this function can be used with functions ``f`` with PyTree inputs and output.
    Abusing notation, for a PyTree ``pt``, let ``pt[idx]`` denote ``tree_map(pt, lambda a: a[idx])``
    (i.e. ``pt[idx]`` denotes the PyTree when indexing all leaves with ``idx``). Let ``a1``, ...,
    ``am`` denote PyTrees whose leaves are all arrays with dimension at least 1, and within each
    PyTree, all leaves have the same length. It holds that
    ``grid_map(f, a1, ..., am)[idx1, ..., idxm] == f(a1[idx1], ..., am[idxm])``, assuming the
    evaluation ``f(a1[idx1], ..., am[idxm])`` is well-defined.

    In addition to this, the arguments ``devices`` and ``max_vmap_size`` enable configuration of
    parallelization and vectorization. ``devices`` specify the list of JAX-visible devices to
    parallelize over, with the calls to ``f`` being evenly split across devices. Within calls to a
    single device, ``max_vmap_size`` controls the number of calls to ``f`` that are executed
    simultaneously using vectorization. All function evaluations are executed in a serial loop, in
    chunks of size ``max_vmap_size * len(devices)``, with a final iteration of size
    ``k * len(devices)`` for some ``k < len(devices)``.

    Finally, the arguments ``key`` and ``keys_per_grid_point`` provide the option to supply every
    call to ``f`` with a randomly generated JAX ``key``, used for pseudo-random number generation in
    ``f``. If ``key`` is specified, it is assumed that the signature of ``f`` is of the form
    ``f(*args, key)``, i.e. the random key is consumed as the last argument, and ``args`` are the
    standard arguments of ``f`` being mapped over. The keys provided to ``f`` are generated
    pseudo-randomly from the ``key`` provided to ``grid_map``. ``keys_per_grid_point`` controls how
    many times ``f`` is evaluated for a given set of deterministic ``args``. If
    ``keys_per_grid_point == 1``, the output of ``grid_map`` will have the same format as described
    above, except that ``f`` will have been provided with a random key for each evaluation. If
    ``keys_per_grid_point > 1``, an additional axis will be added to the output arrays indexing
    repeated evaluation of the function for a fixed value of the deterministic arguments, but for
    different keys. Lastly, the ``key`` argument of ``f`` is assumed to be a JAX-compatible
    argument.

    Notes:
    * This function is a convenience wrapper around JAX's ``xmap`` transformation.
    * The ``nonjax_argnums`` and ``nonjax_argnames`` arguments can be used to prevent JAX mapping
      over a subset of the arguments. If these are used, a normal python loop will be used to map
      over the product of these arguments, and the remaining arguments will be mapped over using
      JAX's mapping functionality. As such, parallelization will only be utilized for the remaining
      arguments. Note that the "non-JAX" arguments specified by ``nonjax_argnums`` and
      ``nonjax_argnames`` are assumed to be standard iterators over the different values of the
      arguments (in contrast to the PyTree structure of JAX-compatible arguments.) Note, however,
      that the output of ``f`` is still assumed to output a PyTree with consistent shape across
      all argument values.

    Args:
        f: The function to map.
        *args: A tuple of PyTrees. Should be the same length as the number of arguments to ``f``.
        devices: 1d numpy object array of devices to parallelize over. Defaults to
            ``np.array(jax.devices())``.
        max_vmap_size: The maximum number of inputs to vectorize over within a device. If the first
            device type is CPU, this will default to ``1``, and if GPU, will default to
            ``len(input_array) / len(devices)``.
        nonjax_argnums: Positional arguments to not map over.
        nonjax_argnames: Named arguments to not map over.
        key: A JAX key to be used for generating randomness. See the function doc string for
            how this impacts the behaviour.
        keys_per_grid_point: If ``key is not None``, controls the number of times ``f`` is
            evaluated with a random key per the rest of the inputs.
    Returns:
        PyTree containing ``f`` evaluated on all combinations of inputs.
    Raises:
        QiskitError: If ``devices`` is of invalid shape.
    """

    if devices is None:
        devices = np.array(jax.devices())
    elif not devices.ndim == 1:
        raise QiskitError("devices must be a 1d array.")

    if (nonjax_argnums is None) and (nonjax_argnames is None):
        # take product of args and map over leading axis
        if key is None:
            args_product = tree_product(args)
        else:
            args_product = _tree_product_with_keys(
                args, key=key, keys_per_grid_point=keys_per_grid_point
            )

        output_1d = _1d_map(f, *args_product, devices=devices, max_vmap_size=max_vmap_size)

        # reshape first axis and return result
        map_shape = tuple(len(tree_flatten(arg)[0][0]) for arg in args)

        # add an extra dimension if more than one key per input was used
        if key is not None and keys_per_grid_point > 1:
            map_shape = map_shape + (keys_per_grid_point,)

        return tree_map(lambda x: x.reshape(map_shape + x.shape[1:]), output_1d)

    if nonjax_argnums is None:
        nonjax_argnums = []
    else:
        for idx in nonjax_argnums:
            if not isinstance(idx, int):
                raise QiskitError("All entries in nonjax_argnums must be ints.")

    # convert argnames to argnums
    if nonjax_argnames is not None:
        all_argnames = inspect.getfullargspec(f).args
        new_argnums = [all_argnames.index(name) for name in nonjax_argnames]
        nonjax_argnums = nonjax_argnums + new_argnums

    # get unique argnums and sort them
    nonjax_argnums = list(set(nonjax_argnums))
    nonjax_argnums.sort()

    # redefined function with nonjax args moved to the front
    g = _move_args_to_front(f, nonjax_argnums)

    nonjax_args = []
    dynamic_args = []
    for idx, arg in enumerate(args):
        if idx in nonjax_argnums:
            nonjax_args.append(arg)
        else:
            dynamic_args.append(arg)
    nonjax_args = tuple(nonjax_args)
    dynamic_args = tuple(dynamic_args)

    nonjax_args_product = product(*nonjax_args)

    # setup dynamic_args_product depending on of randomness is involved
    if key is not None:
        num_nonjax_combos = np.prod(tuple(len(arg) for arg in nonjax_args))
        keys = jax.random.split(key, num_nonjax_combos)
        dynamic_args_product = _tree_product_with_keys(dynamic_args, keys[0], keys_per_grid_point)

        # used to later replace keys without taking whole product
        num_keys_per_map = (
            np.prod(tuple(len(tree_flatten(arg)[0][0]) for arg in dynamic_args))
            * keys_per_grid_point
        )
    else:
        dynamic_args_product = tree_product(dynamic_args)

    outputs = []
    for idx, current_nonjax_args in enumerate(nonjax_args_product):

        if key is not None and idx > 0:
            dynamic_args_product = dynamic_args_product[:-1] + (
                jax.random.split(keys[idx], num_keys_per_map),
            )

        outputs.append(
            _1d_map(
                partial(g, *current_nonjax_args),
                *dynamic_args_product,
                devices=devices,
                max_vmap_size=max_vmap_size,
            )
        )

    output_1d = tree_concatenate(jax.device_put(outputs, devices[0]))

    # reshape first axis to be multidimensional with the arguments in the nonjax + dynamic order
    map_shape = tuple(len(arg) for arg in nonjax_args) + tuple(
        len(tree_flatten(arg)[0][0]) for arg in dynamic_args
    )

    # if keys_per_grid_point > 1 add a further dimension
    if key is not None and keys_per_grid_point > 1:
        map_shape = map_shape + (keys_per_grid_point,)

    # reshape based on input shapes
    reshaped_output = tree_map(lambda x: x.reshape(map_shape + x.shape[1:]), output_1d)

    # reorder first axes to correspond to the original argument order
    num_args = len(args) if (key is None or keys_per_grid_point == 1) else len(args) + 1
    current_arg_order = nonjax_argnums + list(
        idx for idx in range(num_args) if idx not in nonjax_argnums
    )
    original_arg_location = [current_arg_order.index(idx) for idx in range(num_args)]

    def axis_reorder(x):
        x_axis_order = original_arg_location + list(range(num_args, x.ndim))
        return x.transpose(x_axis_order)

    return tree_map(axis_reorder, reshaped_output)


def _1d_map(
    f: Callable,
    *args: Tuple["PyTree"],
    devices: Optional[np.array] = None,
    max_vmap_size: Optional[int] = None,
) -> jnp.array:
    """Map f over the leading axis of args (assumed to be PyTrees) using a combination of device
    parallelization and vectorization.

    Implicit in this mapping is the assumption that all leaves are arrays that have at least one
    dimension and have the same length.

    The mapping is parallelized over ``devices`` in chunks of ``vmap_size`` per device. Each chunk
    of size ``vmap_size`` passed to a single device will be evaluated via vectorization. This is a
    convenience wrapper over the ``xmap`` transformation in JAX.

    Args:
        f: The function to map, assumed to be a function of a single array.
        *args: The arguments to map ``f`` over.
        devices: 1d numpy object array of devices to parallelize over. Defaults to
            ``np.array(jax.devices())``.
        max_vmap_size: The maximum number of inputs to vectorize over within a device. If the first
            device type is CPU, this will default to ``1``, and if GPU, will default to
            ``len(input_array) / len(devices)``.
    Returns:
        ``f`` mapped over the leading axis of ``input_array``.
    Raises:
        QiskitError: If devices are of invalid shape.
    """

    if devices is None:
        devices = np.array(jax.devices())
    elif not devices.ndim == 1:
        raise QiskitError("devices must be a 1d array.")

    # we should be able to rewrite everything after this using a single evaluation of xmap_f by
    # utilizing SerialLoop, but it's currently raising errors when used with odeint
    xmap_f = xmap(
        f,
        in_axes={0: "a"},
        out_axes={0: "a"},
        axis_resources={"a": ("x",)},
    )

    # get number of inputs being mapped over
    axis_size = len(tree_flatten(args[0])[0][0])

    # set max_vmap_size based on device type
    if max_vmap_size is None:
        if devices[0].platform == "cpu":
            max_vmap_size = 1
        else:
            max_vmap_size = int(axis_size / len(devices))

    def input_index(start_idx, end_idx):
        return tree_map(lambda x: x[start_idx:end_idx], args)

    # iterate in chunks
    outputs = []
    current_idx = 0
    while current_idx < axis_size:
        num_evals_remaining = axis_size - current_idx
        last_idx = current_idx
        # if there are more evaluations remaining than there are devices, evaluate
        if num_evals_remaining > len(devices):
            vmap_size = min(int((axis_size - current_idx) / len(devices)), max_vmap_size)
            current_idx = last_idx + vmap_size * len(devices)
            with Mesh(devices, ("x",)):
                outputs.append(xmap_f(*input_index(last_idx, current_idx)))
        else:
            current_idx = last_idx + num_evals_remaining
            with Mesh(devices[:num_evals_remaining], ("x",)):
                outputs.append(xmap_f(*input_index(last_idx, current_idx)))

    # combine and return outcomes
    return tree_concatenate(jax.device_put(outputs, devices[0]))


def _move_args_to_front(f, argnums):
    """Define a new function ``g`` giving the same output as ``f``, but with the positional args
    whose locations are given by ``argnums`` moved to the beginning of ``g``. ``argnums`` is assumed
    to be a sorted list of integers.
    """

    def g(*args):
        f_args = list(args[len(argnums) :])
        for idx, arg in zip(argnums, args[: len(argnums)]):
            f_args.insert(idx, arg)

        return f(*f_args)

    return g


def _tree_product_with_keys(trees, key: jnp.ndarray, keys_per_grid_point: int = 1):

    # take args product with a placeholder for proper structure
    key_placeholder = jnp.array([0] * keys_per_grid_point)
    args_product = tree_product(trees + (key_placeholder,))

    # generate an array of keys
    num_keys_needed = len(tree_flatten(args_product)[0][0])
    keys = jax.random.split(key, num_keys_needed)

    # replace placeholder with actual keys
    return args_product[:-1] + (keys,)
