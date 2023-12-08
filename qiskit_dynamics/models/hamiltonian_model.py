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
Hamiltonian models module.
"""

from typing import Union, List, Optional
import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import norm as spnorm

from qiskit import QiskitError
from qiskit.quantum_info.operators import Operator

from qiskit_dynamics import DYNAMICS_NUMPY as unp
from qiskit_dynamics import DYNAMICS_NUMPY_ALIAS as numpy_alias
from qiskit_dynamics.arraylias.alias import ArrayLike
from qiskit_dynamics.array import Array
from qiskit_dynamics.signals import Signal, SignalList
from qiskit_dynamics.type_utils import to_numeric_matrix_type, to_array
from .generator_model import (
    GeneratorModel,
    _static_operator_into_frame_basis,
    _operators_into_frame_basis
)
from .rotating_frame import RotatingFrame


class HamiltonianModel(GeneratorModel):
    r"""A model of a Hamiltonian for the Schrodinger equation.

    This class represents a Hamiltonian as a time-dependent decomposition the form:

    .. math::
        H(t) = H_d + \sum_j s_j(t) H_j,

    where :math:`H_j` are Hermitian operators, :math:`H_d` is the static component, and the
    :math:`s_j(t)` are either :class:`~qiskit_dynamics.signals.Signal` objects or numerical
    constants. Constructing a :class:`~qiskit_dynamics.models.HamiltonianModel` requires specifying
    the above decomposition, e.g.:

    .. code-block:: python

        hamiltonian = HamiltonianModel(static_operator=static_operator,
                                       operators=operators,
                                       signals=signals)

    This class inherits most functionality from :class:`GeneratorModel`, with the following
    modifications:

        * The operators :math:`H_d` and :math:`H_j` are assumed and verified to be Hermitian.
        * Rotating frames are dealt with assuming the structure of the Schrodinger equation. I.e.
          Evaluating the Hamiltonian :math:`H(t)` in a frame :math:`F = -iH_0`, evaluates the
          expression :math:`e^{-tF}H(t)e^{tF} - H_0`.
    """

    def __init__(
        self,
        static_operator: Optional[Array] = None,
        operators: Optional[List[Operator]] = None,
        signals: Optional[Union[SignalList, List[Signal]]] = None,
        rotating_frame: Optional[Union[Operator, Array, RotatingFrame]] = None,
        in_frame_basis: bool = False,
        array_library: Optional[str] = None,
        validate: bool = True,
    ):
        """Initialize, ensuring that the operators are Hermitian.

        Args:
            static_operator: Time-independent term in the Hamiltonian.
            operators: List of Operator objects.
            signals: List of coefficients :math:`s_i(t)`. Not required at instantiation, but
                necessary for evaluation of the model.
            rotating_frame: Rotating frame operator. If specified with a 1d array, it is interpreted
                as the diagonal of a diagonal matrix. Assumed to store the antihermitian matrix
                F = -iH.
            in_frame_basis: Whether to represent the model in the basis in which the rotating
                frame operator is diagonalized.
            array_library: Array library for storing the operators in the model. Supported options
                are ``'numpy'``, ``'jax'``, ``'jax_sparse'``, and ``'scipy_sparse'``. If ``None``,
                the arrays will be handled by general dispatching rules. Call
                ``help(GeneratorModel.array_library)`` for more details.
            validate: If ``True`` check input operators are Hermitian. Note that this is
                incompatible with JAX transformations.

        Raises:
            QiskitError: if operators are not Hermitian
        """

        # verify operators are Hermitian, and if so instantiate
        if validate:
            if (operators is not None) and (not all(is_hermitian(numpy_alias(like=array_library).asarray(op)) for op in operators)):
                raise QiskitError("""HamiltonianModel operators must be Hermitian.""")
            if (static_operator is not None) and (not is_hermitian(numpy_alias(like=array_library).asarray(static_operator))):
                raise QiskitError("""HamiltonianModel static_operator must be Hermitian.""")

        super().__init__(
            static_operator=static_operator,
            operators=operators,
            signals=signals,
            rotating_frame=rotating_frame,
            in_frame_basis=in_frame_basis,
            array_library=array_library,
        )

    @property
    def rotating_frame(self) -> RotatingFrame:
        """The rotating frame.

        This property can be set with a :class:`RotatingFrame` instance, or any valid constructor
        argument to this class. It can either be Hermitian or anti-Hermitian, and in either case
        only the anti-Hermitian version :math:`F=-iH` will be stored.

        Setting this property updates the internal operator list to use the new frame.
        """
        return super().rotating_frame

    @rotating_frame.setter
    def rotating_frame(self, rotating_frame: Union[Operator, Array, RotatingFrame]) -> Array:
        new_frame = RotatingFrame(rotating_frame)

        # convert static operator to new frame setup
        static_op = self._get_static_operator(in_frame_basis=True)
        if static_op is not None:
            static_op = -1j * static_op

        new_static_operator = transfer_static_operator_between_frames(
            static_op,
            new_frame=new_frame,
            old_frame=self.rotating_frame,
        )

        if new_static_operator is not None:
            new_static_operator = 1j * new_static_operator

        # convert operators to new frame set up
        new_operators = transfer_operators_between_frames(
            self._get_operators(in_frame_basis=True),
            new_frame=new_frame,
            old_frame=self.rotating_frame,
        )

        self._rotating_frame = new_frame

        self._operator_collection = construct_operator_collection(
            self.evaluation_mode, new_static_operator, new_operators
        )

    def evaluate(self, time: float) -> Array:
        """Evaluate the model in array format as a matrix, independent of state.

        Args:
            time: The time to evaluate the model at.

        Returns:
            Array: The evaluated model as an anti-Hermitian matrix.

        Raises:
            QiskitError: If model cannot be evaluated.
        """
        return -1j * super().evaluate(time)

    def evaluate_rhs(self, time: float, y: Array) -> Array:
        r"""Evaluate ``-1j * H(t) @ y``.

        Args:
            time: The time to evaluate the model at .
            y: Array specifying system state.

        Returns:
            Array defined by :math:`G(t) \times y`.

        Raises:
            QiskitError: If model cannot be evaluated.
        """
        return -1j * super().evaluate_rhs(time, y)


def is_hermitian(
    operator: ArrayLike, tol: Optional[float] = 1e-10
) -> bool:
    """Validate that an operator is Hermitian.

    Args:
        operators: A 2d array representing a single operator.
        tol: Tolerance for checking zeros.

    Returns:
        bool: Whether or not the operator is Hermitian to within tolerance.

    Raises:
        QiskitError: If an unexpeted type is received.
    """
    if issparse(operator):
        return spnorm(operator - operator.conj().transpose()) < tol
    elif type(operator).__name__ == "BCOO":
        # fall back on array case for BCOO
        return is_hermitian(operator.todense())
    elif isinstance(operator, ArrayLike):
        adj = None
        adj = np.transpose(np.conjugate(operator))
        return np.linalg.norm(adj - operator) < tol

    raise QiskitError("is_hermitian got an unexpected type.")
