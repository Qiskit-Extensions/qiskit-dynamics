# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may obtain a copy of this license
# in the LICENSE.txt file in the root directory of this source tree or at
# http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this copyright notice, and modified
# files need to carry a notice indicating that they have been altered from the originals.
# pylint: disable=invalid-name

# Classes that don't explicitly inherit QiskitDynamicsTestCase get no-member errors
# pylint: disable=no-member

"""Tests for generator_models.py. """

from scipy.sparse import issparse, csr_matrix
import numpy as np
import numpy.random as rand
from qiskit import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit_dynamics.models import GeneratorModel, RotatingFrame
from qiskit_dynamics.models.generator_model import (
    transfer_static_operator_between_frames,
    transfer_operators_between_frames,
)
from qiskit_dynamics.signals import Signal, SignalList
from qiskit_dynamics.arraylias import ArrayLike
from qiskit_dynamics.arraylias import DYNAMICS_NUMPY as unp
from qiskit_dynamics.arraylias import DYNAMICS_SCIPY as scp
from qiskit_dynamics.arraylias.alias import _numpy_multi_dispatch
from ..common import QiskitDynamicsTestCase, test_array_backends


try:
    from jax import jit
    import jax.numpy as jnp
except ImportError:
    pass


class TestGeneratorModelErrors(QiskitDynamicsTestCase):
    """Test deliberate error modes."""

    def test_both_static_operator_operators_None(self):
        """Test errors raising for a mal-formed GeneratorModel."""

        with self.assertRaisesRegex(QiskitError, "at least one of static_operator or operators"):
            GeneratorModel(static_operator=None, operators=None)

    def test_operators_None_signals_not_None(self):
        """Test setting signals with operators being None."""

        with self.assertRaisesRegex(QiskitError, "Signals must be None if operators is None."):
            GeneratorModel(
                static_operator=np.array([[1.0, 0.0], [0.0, -1.0]]), operators=None, signals=[1.0]
            )

        # test after initial instantiation
        model = GeneratorModel(static_operator=np.array([[1.0, 0.0], [0.0, -1.0]]))
        with self.assertRaisesRegex(QiskitError, "Signals must be None if operators is None."):
            model.signals = [1.0]

    def test_operators_signals_length_mismatch(self):
        """Test setting operators and signals to incompatible lengths."""
        with self.assertRaisesRegex(QiskitError, "same length as operators."):
            GeneratorModel(operators=np.array([[[1.0, 0.0], [0.0, -1.0]]]), signals=[1.0, 1.0])

        # test after initial instantiation
        model = GeneratorModel(operators=np.array([[[1.0, 0.0], [0.0, -1.0]]]))
        with self.assertRaisesRegex(QiskitError, "same length as operators."):
            model.signals = [1.0, 1.0]

    def test_signals_bad_format(self):
        """Test setting signals in an unacceptable format."""
        with self.assertRaisesRegex(QiskitError, "unaccepted format."):
            GeneratorModel(operators=np.array([[[1.0, 0.0], [0.0, -1.0]]]), signals=lambda t: t)

        # test after initial instantiation
        model = GeneratorModel(operators=np.array([[[1.0, 0.0], [0.0, -1.0]]]))
        with self.assertRaisesRegex(QiskitError, "unaccepted format."):
            model.signals = lambda t: t


class TestGeneratorModel:
    """Tests for GeneratorModel."""

    def setUp(self):
        """Setup GeneratorModel"""
        self.X = self.asarray(Operator.from_label("X"))
        self.Y = self.asarray(Operator.from_label("Y"))
        self.Z = self.asarray(Operator.from_label("Z"))

        # define a basic model
        w = 2.0
        r = 0.5
        operators = [-1j * 2 * np.pi * self.Z / 2, -1j * 2 * np.pi * r * self.X / 2]
        signals = [w, Signal(1.0, w)]

        self.w = 2
        self.r = r
        self.basic_model = GeneratorModel(operators=operators, signals=signals)

    def test_diag_frame_operator_basic_model(self):
        """Test setting a diagonal frame operator for the internally
        set up basic model.
        """

        self._basic_frame_evaluate_test(self.asarray([1j, -1j]), 1.123)
        self._basic_frame_evaluate_test(self.asarray([1j, -1j]), np.pi)

    def test_non_diag_frame_operator_basic_model(self):
        """Test setting a non-diagonal frame operator for the internally
        set up basic model.
        """
        self._basic_frame_evaluate_test(-1j * (self.Y + self.Z), 1.123)
        self._basic_frame_evaluate_test(-1j * (self.Y - self.Z), np.pi)

    def _basic_frame_evaluate_test(self, frame_operator, t):
        """Routine for testing setting of valid frame operators using the
        basic_model.
        """
        self.basic_model.rotating_frame = frame_operator

        # convert to 2d array
        if isinstance(frame_operator, Operator):
            frame_operator = self.asarray(frame_operator)
        if isinstance(frame_operator, ArrayLike) and frame_operator.ndim == 1:
            frame_operator = unp.diag(frame_operator)

        value = self.basic_model(t)

        i2pi = -1j * 2 * np.pi

        U = scp.linalg.expm(-np.array(frame_operator) * t)

        # drive coefficient
        d_coeff = self.r * np.cos(2 * np.pi * self.w * t)

        # manually evaluate frame
        expected = (
            i2pi * self.w * U @ self.Z @ U.conj().transpose() / 2
            + d_coeff * i2pi * U @ self.X @ U.conj().transpose() / 2
            - frame_operator
        )

        self.assertAllClose(value, expected)

    def test_evaluate_no_frame_basic_model(self):
        """Test evaluation without a frame in the basic model."""

        t = 3.21412
        value = self.basic_model(t)
        i2pi = -1j * 2 * np.pi
        d_coeff = self.r * np.cos(2 * np.pi * self.w * t)
        expected = i2pi * self.w * self.Z / 2 + i2pi * d_coeff * self.X / 2

        self.assertAllClose(value, expected)

    def test_evaluate_generator_in_frame_basis_basic_model(self):
        """Test generator evaluation in frame basis in the basic_model."""

        frame_op = -1j * self.asarray(self.X + 0.2 * self.Y + 0.1 * self.Z)

        # enter the frame given by the -1j * X
        self.basic_model.rotating_frame = frame_op

        # set to operate in frame basis
        self.basic_model.in_frame_basis = True

        # get the frame basis that is used in model
        _, U = unp.linalg.eigh(1j * frame_op)

        t = 3.21412
        value = self.basic_model(t)

        # compose the frame basis transformation with the exponential
        # frame rotation (this will be multiplied on the right)
        U = scp.linalg.expm(unp.asarray(frame_op) * t) @ U
        Uadj = U.conj().transpose()

        i2pi = -1j * 2 * np.pi
        d_coeff = self.r * np.cos(2 * np.pi * self.w * t)
        expected = Uadj @ (i2pi * self.w * self.Z / 2 + i2pi * d_coeff * self.X / 2 - frame_op) @ U

        self.assertAllClose(value, expected)

    def test_evaluate_pseudorandom(self):
        """Test evaluate with pseudorandom inputs."""

        rng = np.random.default_rng(30493)
        num_terms = 3
        dim = 5
        b = 1.0  # bound on size of random terms
        rand_op = rng.uniform(low=-b, high=b, size=(dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(dim, dim)
        )
        frame_op = self.asarray(rand_op - rand_op.conj().transpose())
        randoperators = rng.uniform(low=-b, high=b, size=(num_terms, dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(num_terms, dim, dim)
        )

        rand_coeffs = self.asarray(
            rng.uniform(low=-b, high=b, size=(num_terms))
            + 1j * rng.uniform(low=-b, high=b, size=(num_terms))
        )
        rand_carriers = self.asarray(rng.uniform(low=-b, high=b, size=(num_terms)))
        rand_phases = self.asarray(rng.uniform(low=-b, high=b, size=(num_terms)))

        self._test_evaluate(frame_op, randoperators, rand_coeffs, rand_carriers, rand_phases)

        rng = np.random.default_rng(94818)
        num_terms = 5
        dim = 10
        b = 1.0  # bound on size of random terms
        rand_op = rng.uniform(low=-b, high=b, size=(dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(dim, dim)
        )
        frame_op = self.asarray(rand_op - rand_op.conj().transpose())
        randoperators = self.asarray(
            rng.uniform(low=-b, high=b, size=(num_terms, dim, dim))
            + 1j * rng.uniform(low=-b, high=b, size=(num_terms, dim, dim))
        )

        rand_coeffs = self.asarray(
            rng.uniform(low=-b, high=b, size=(num_terms))
            + 1j * rng.uniform(low=-b, high=b, size=(num_terms))
        )
        rand_carriers = self.asarray(rng.uniform(low=-b, high=b, size=(num_terms)))
        rand_phases = self.asarray(rng.uniform(low=-b, high=b, size=(num_terms)))
        self._test_evaluate(frame_op, randoperators, rand_coeffs, rand_carriers, rand_phases)

    def _test_evaluate(self, frame_op, operators, coefficients, carriers, phases):
        """Test evaluation of both dense and sparse."""
        sig_list = []
        for coeff, freq, phase in zip(coefficients, carriers, phases):

            def get_env_func(coeff=coeff):
                # pylint: disable=unused-argument
                def env(t):
                    return coeff

                return env

            sig_list.append(Signal(get_env_func(), freq, phase))
        model = GeneratorModel(operators=operators, signals=sig_list)
        model.rotating_frame = frame_op

        value = model(1.0)
        coeffs = unp.real(coefficients * unp.exp(1j * 2 * np.pi * carriers * 1.0 + 1j * phases))

        self.assertAllClose(
            model.rotating_frame.operator_out_of_frame_basis(
                model._operator_collection.evaluate(coeffs)
            ),
            _numpy_multi_dispatch(coeffs, operators, path="tensordot", axes=1) - frame_op,
        )
        expected = (
            scp.linalg.expm(-unp.asarray(frame_op))
            @ _numpy_multi_dispatch(coeffs, operators, path="tensordot", axes=1)
            @ scp.linalg.expm(unp.asarray(frame_op))
            - frame_op
        )

        self.assertAllClose(value, expected)
        if self.array_library != "jax":
            model.evaluation_mode = "sparse"
            state = np.arange(operators.shape[-1] * 4).reshape(operators.shape[-1], 4)
            value = model(1.0)
            if issparse(value):
                value = value.toarray()
            self.assertAllClose(value, expected)

            val_with_state = model(1.0, state)
            self.assertAllClose(val_with_state, value @ state)

    def test_signal_setting(self):
        """Test updating the signals."""

        signals = [Signal(lambda t: 2 * t, 1.0), Signal(lambda t: t**2, 2.0)]
        self.basic_model.signals = signals

        t = 0.1
        value = self.basic_model(t)
        i2pi = -1j * 2 * np.pi
        Z_coeff = (2 * t) * np.cos(2 * np.pi * 1 * t)
        X_coeff = self.r * (t**2) * np.cos(2 * np.pi * 2 * t)
        expected = i2pi * Z_coeff * self.Z / 2 + i2pi * X_coeff * self.X / 2
        self.assertAllClose(value, expected)

    def test_signal_setting_None(self):
        """Test setting signals to None"""

        self.basic_model.signals = None
        self.assertTrue(self.basic_model.signals is None)

    def test_evaluate_analytic(self):
        """Test for checking that with known operators that
        the Model returns the analyticlly known values."""

        test_operator_list = self.asarray([self.X, self.Y, self.Z])
        signals = SignalList([Signal(1, j / 3) for j in range(3)])
        simple_model = GeneratorModel(
            operators=test_operator_list, static_operator=None, signals=signals, rotating_frame=None
        )

        res = simple_model.evaluate(2)
        self.assertAllClose(res, self.asarray([[-0.5 + 0j, 1.0 + 0.5j], [1.0 - 0.5j, 0.5 + 0j]]))

        simple_model.static_operator = np.eye(2)
        res = simple_model.evaluate(2)
        self.assertAllClose(res, self.asarray([[0.5 + 0j, 1.0 + 0.5j], [1.0 - 0.5j, 1.5 + 0j]]))

    def test_evaluate_in_frame_basis_analytic(self):
        """Tests evaluation in rotating frame against analytic values,
        both in specified basis and in frame basis."""
        test_operator_list = self.asarray([self.X, self.Y, self.Z])
        signals = SignalList([Signal(1, j / 3) for j in range(3)])
        simple_model = GeneratorModel(
            operators=test_operator_list, static_operator=None, signals=signals, rotating_frame=None
        )
        simple_model.static_operator = np.eye(2)
        fop = self.asarray([[0, 1j], [1j, 0]])
        simple_model.rotating_frame = fop
        res = simple_model(2.0)
        expected = (
            scp.linalg.expm(unp.asarray(-2 * fop))
            @ (self.asarray([[0.5 + 0j, 1.0 + 0.5j], [1.0 - 0.5j, 1.5 + 0j]]) - fop)
            @ scp.linalg.expm(unp.asarray(2 * fop))
        )
        self.assertAllClose(res, expected)

        simple_model.in_frame_basis = True
        res = simple_model(2)
        expected = (
            unp.asarray([[1, 1], [1, -1]])
            / np.sqrt(2)
            @ expected
            @ unp.asarray([[1, 1], [1, -1]])
            / np.sqrt(2)
        )

        self.assertAllClose(res, expected)

    def test_order_of_assigning_properties(self):
        """Tests whether setting the frame, static_operator, and signals
        in the constructor is the same as constructing without
        and then assigning them in any order.
        """

        paulis = self.asarray([self.X, self.Y, self.Z])
        extra = self.asarray(np.eye(2))
        state = self.asarray([0.2, 0.5])

        t = 2

        sarr = [Signal(1, j / 3) for j in range(3)]
        sigvals = SignalList(sarr)(t)

        farr = self.asarray(np.array([[3j, 2j], [2j, 0]]))
        farr2 = self.asarray(np.array([[1j, 2], [-2, 3j]]))
        evals, evect = unp.linalg.eig(farr)
        diafarr = unp.diag(evals)

        paulis_in_frame_basis = unp.conjugate(unp.transpose(evect)) @ paulis @ evect

        ## Run checks without rotating frame for now
        gm1 = GeneratorModel(operators=paulis, signals=sarr, static_operator=extra)
        gm2 = GeneratorModel(operators=paulis, static_operator=extra)
        gm2.signals = sarr
        gm3 = GeneratorModel(operators=paulis, static_operator=extra)
        gm3.signals = SignalList(sarr)

        # All should be the same, because there are no frames involved
        t11 = gm1.evaluate(t)
        gm1.in_frame_basis = True
        t12 = gm1.evaluate(t)
        t21 = gm2.evaluate(t)
        gm2.in_frame_basis = True
        t22 = gm2.evaluate(t)
        t31 = gm3.evaluate(t)
        gm3.in_frame_basis = True
        t32 = gm3.evaluate(t)
        t_analytical = self.asarray([[0.5, 1.0 + 0.5j], [1.0 - 0.5j, 1.5]])

        self.assertAllClose(t11, t12)
        self.assertAllClose(t11, t21)
        self.assertAllClose(t11, t22)
        self.assertAllClose(t11, t31)
        self.assertAllClose(t11, t32)
        self.assertAllClose(t11, t_analytical)

        # now work with a specific statevector
        ts1 = gm1.evaluate_rhs(t, state)
        gm1.in_frame_basis = True
        ts2 = gm1.evaluate_rhs(t, state)
        ts_analytical = self.asarray([0.6 + 0.25j, 0.95 - 0.1j])

        self.assertAllClose(ts1, ts2)
        self.assertAllClose(ts1, ts_analytical)
        self.assertAllClose(gm1(t) @ state, ts1)
        gm1.in_frame_basis = False
        self.assertAllClose(gm1(t) @ state, ts1)

        ## Now, run checks with frame
        # If passing a frame in the first place, operators must be in frame basis.abs
        # Testing at the same time whether having static_operatorrift = None is an issue.
        gm1 = GeneratorModel(
            operators=paulis, signals=sarr, rotating_frame=RotatingFrame(farr), in_frame_basis=True
        )
        gm2 = GeneratorModel(operators=paulis, rotating_frame=farr, in_frame_basis=True)
        gm2.signals = SignalList(sarr)
        gm3 = GeneratorModel(operators=paulis, rotating_frame=farr, in_frame_basis=True)
        gm3.signals = sarr
        # Does adding a frame after make a difference?
        # If so, does it make a difference if we add signals or the frame first?
        gm4 = GeneratorModel(operators=paulis, in_frame_basis=True)
        gm4.signals = sarr
        gm4.rotating_frame = farr
        gm5 = GeneratorModel(operators=paulis, in_frame_basis=True)
        gm5.rotating_frame = farr
        gm5.signals = sarr
        gm6 = GeneratorModel(operators=paulis, signals=sarr, in_frame_basis=True)
        gm6.rotating_frame = farr
        # If we go to one frame, then transform back, does this make a difference?
        gm7 = GeneratorModel(operators=paulis, signals=sarr, in_frame_basis=True)
        gm7.rotating_frame = farr2
        gm7.rotating_frame = farr

        t_in_frame_actual = self.asarray(
            unp.diag(unp.exp(-t * evals))
            @ (
                _numpy_multi_dispatch(sigvals, paulis_in_frame_basis, path="tensordot", axes=1)
                - diafarr
            )
            @ unp.diag(unp.exp(t * evals))
        )
        tf1 = gm1.evaluate(t)
        tf2 = gm2.evaluate(t)
        tf3 = gm3.evaluate(t)
        tf4 = gm4.evaluate(t)
        tf5 = gm5.evaluate(t)
        tf6 = gm6.evaluate(t)
        tf7 = gm7.evaluate(t)

        self.assertAllClose(t_in_frame_actual, tf1)
        self.assertAllClose(tf1, tf2)
        self.assertAllClose(tf1, tf3)
        self.assertAllClose(tf1, tf4)
        self.assertAllClose(tf1, tf5)
        self.assertAllClose(tf1, tf6)
        self.assertAllClose(tf1, tf7)

    def test_evaluate_rhs_lmult_equivalent_analytic(self):
        """Tests whether evaluate(t) @ state == evaluate_rhs(t,state)
        for analytically known values."""

        paulis = self.asarray([self.X, self.Y, self.Z])
        extra = self.asarray(np.eye(2))

        t = 2

        farr = self.asarray(np.array([[3j, 2j], [2j, 0]]))
        evals, evect = unp.linalg.eig(farr)
        diafarr = unp.diag(evals)

        paulis_in_frame_basis = unp.conjugate(unp.transpose(evect)) @ paulis @ evect
        extra_in_basis = evect.T.conj() @ extra @ evect

        sarr = [Signal(1, j / 3) for j in range(3)]
        sigvals = SignalList(sarr)(t)

        t_in_frame_actual = self.asarray(
            unp.diag(unp.exp(-t * evals))
            @ (
                _numpy_multi_dispatch(sigvals, paulis_in_frame_basis, path="tensordot", axes=1)
                + extra_in_basis
                - diafarr
            )
            @ unp.diag(unp.exp(t * evals))
        )

        state = self.asarray([0.3, 0.1])
        state_in_frame_basis = unp.conjugate(unp.transpose(evect)) @ state

        gm1 = GeneratorModel(
            operators=paulis,
            signals=sarr,
            rotating_frame=farr,
            static_operator=extra,
            in_frame_basis=True,
        )
        self.assertAllClose(gm1(t), t_in_frame_actual)
        self.assertAllClose(
            gm1(t, state_in_frame_basis),
            t_in_frame_actual @ state_in_frame_basis,
        )

        t_not_in_frame_actual = self.asarray(
            scp.linalg.expm(unp.asarray(-t * farr))
            @ (_numpy_multi_dispatch(sigvals, paulis, path="tensordot", axes=1) + extra - farr)
            @ scp.linalg.expm(unp.asarray(t * farr))
        )

        gm2 = GeneratorModel(
            operators=paulis,
            signals=sarr,
            rotating_frame=farr,
            static_operator=extra,
            in_frame_basis=False,
        )
        gm1.in_frame_basis = False
        self.assertAllClose(gm2(t), t_not_in_frame_actual)
        self.assertAllClose(gm1(t, state), t_not_in_frame_actual @ state)

        # now, remove the frame
        gm1.rotating_frame = None
        gm2.rotating_frame = None

        t_expected = _numpy_multi_dispatch(sigvals, paulis, path="tensordot", axes=1) + extra

        state_in_frame_basis = state

        self.assertAllClose(gm1._get_operators(True), gm1._get_operators(False))
        self.assertAllClose(gm1._get_static_operator(True), gm1._get_static_operator(False))

        gm1.in_frame_basis = True
        self.assertAllClose(gm1(t), t_expected)
        self.assertAllClose(gm1(t, state), t_expected @ state_in_frame_basis)
        self.assertAllClose(gm2(t), t_expected)
        self.assertAllClose(gm2(t, state), t_expected @ state_in_frame_basis)

    def test_evaluate_rhs_vectorized_pseudorandom(self):
        """Test for whether evaluating a model at m different
        states, each with an n-length statevector, by passing
        an (m,n) Array provides the same value as passing each
        (n) Array individually."""
        rand.seed(9231)
        n = 8
        k = 4
        m = 2
        t = rand.rand()
        sig_list = SignalList([Signal(rand.rand(), rand.rand()) for j in range(k)])
        normal_states = rand.uniform(-1, 1, (n))
        vectorized_states = rand.uniform(-1, 1, (n, m))

        operators = rand.uniform(-1, 1, (k, n, n))

        gm = GeneratorModel(operators=operators, static_operator=None, signals=sig_list)
        self.assertTrue(gm.evaluate_rhs(t, normal_states).shape == (n,))
        self.assertTrue(gm.evaluate_rhs(t, vectorized_states).shape == (n, m))
        for i in range(m):
            self.assertAllClose(
                gm.evaluate_rhs(t, vectorized_states)[:, i],
                gm.evaluate_rhs(t, vectorized_states[:, i]),
            )

        farr = rand.uniform(-1, 1, (n, n))
        farr = farr - np.conjugate(unp.transpose(farr))

        gm.rotating_frame = farr

        self.assertTrue(gm.evaluate_rhs(t, normal_states).shape == (n,))
        self.assertTrue(gm.evaluate_rhs(t, vectorized_states).shape == (n, m))
        vectorized_result = gm.evaluate_rhs(t, vectorized_states)
        for i in range(m):
            self.assertAllClose(
                vectorized_result[:, i],
                gm.evaluate_rhs(t, vectorized_states[:, i]),
            )

        gm.in_frame_basis = True
        vectorized_result = gm(t, vectorized_states)

        self.assertTrue(gm(t, normal_states).shape == (n,))
        self.assertTrue(vectorized_result.shape == (n, m))
        for i in range(m):
            self.assertAllClose(
                vectorized_result[:, i],
                gm(t, vectorized_states[:, i]),
            )

    def test_evaluate_static(self):
        """Test evaluation of a GeneratorModel with only a static component."""

        static_model = GeneratorModel(static_operator=self.X)
        self.assertAllClose(self.X, static_model(1.0))

        # now with frame
        frame_op = -1j * (self.Z + 1.232 * self.Y)
        static_model.rotating_frame = frame_op
        t = 1.2312
        expected = (
            scp.linalg.expm(-frame_op * t) @ (self.X - frame_op) @ scp.linalg.expm(frame_op * t)
        )
        self.assertAllClose(expected, static_model(t))

    def test_get_operators_when_None(self):
        """Test getting operators when None."""

        model = GeneratorModel(static_operator=np.array([[1.0, 0.0], [0.0, -1.0]]))
        self.assertTrue(model.operators is None)


class TestGeneratorModelJax(TestGeneratorModel):
    """Jax version of TestGeneratorModel tests."""

    def test_jitable_funcs(self):
        """Tests whether all functions are jitable.
        Checks if having a frame makes a difference, as well as
        all jax-compatible evaluation_modes."""
        jit(self.basic_model.evaluate)(1.0)
        jit(self.basic_model.evaluate_rhs)(1, jnp.array([0.2, 0.4]))

        self.basic_model.rotating_frame = jnp.array([[3j, 2j], [2j, 0]])

        jit(self.basic_model.evaluate)(1)
        jit(self.basic_model.evaluate_rhs)(1, jnp.array([0.2, 0.4]))

        self.basic_model.rotating_frame = None

    def test_gradable_funcs(self):
        """Tests whether all functions are gradable.
        Checks if having a frame makes a difference, as well as
        all jax-compatible evaluation_modes."""
        self.jit_grad_wrap(self.basic_model.evaluate)(1.0)
        self.jit_grad_wrap(self.basic_model.evaluate_rhs)(1.0, jnp.array([0.2, 0.4]))

        self.basic_model.rotating_frame = jnp.array([[3j, 2j], [2j, 0]])

        self.jit_grad_wrap(self.basic_model.evaluate)(1.0)
        self.jit_grad_wrap(self.basic_model.evaluate_rhs)(1.0, jnp.array([0.2, 0.4]))

        self.basic_model.rotating_frame = None


test_array_backends(TestGeneratorModel, array_libraries=["numpy", "array_nunpy"])
test_array_backends(TestGeneratorModelJax, array_libraries=["jax"])


class TestGeneratorModelSparse:
    """Sparse-mode specific tests."""

    def setUp(self):
        """Setup Pauli operators"""
        self.X = self.asarray(Operator.from_label("X"))
        self.Y = self.asarray(Operator.from_label("Y"))
        self.Z = self.asarray(Operator.from_label("Z"))

    def test_switch_modes_and_evaluate(self):
        """Test construction of a model, switching modes, and evaluating."""

        model = GeneratorModel(static_operator=self.Z, operators=[self.X], signals=[1.0])
        self.assertAllClose(model(1.0), self.Z + self.X)

        model.evaluation_mode = "sparse"
        val = model(1.0)
        self.validate_generator_eval(val, self.Z + self.X)

        model.evaluation_mode = "dense"
        self.assertAllClose(model(1.0), self.Z + self.X)

    def test_frame_change_sparse(self):
        """Test setting a frame after instantiation in sparse mode and evaluating."""
        model = GeneratorModel(
            static_operator=self.Z, operators=[self.X], signals=[1.0], evaluation_mode="sparse"
        )

        # test non-diagonal frame
        model.rotating_frame = self.Z
        expected = (
            scp.linalg.expm(1j * self.Z)
            @ ((1 + 1j) * self.Z + self.X)
            @ scp.linalg.expm(-1j * self.Z)
        )
        val = model(1.0)
        self.assertAllClose(val, expected)

        # test diagonal frame
        model.rotating_frame = np.array([1.0, -1.0])
        val = model(1.0)
        self.validate_generator_eval(val, expected)

    def test_switching_to_sparse_with_frame(self):
        """Test switching to sparse with a frame already set."""

        model = GeneratorModel(
            static_operator=self.Z,
            operators=[self.X],
            signals=[1.0],
            rotating_frame=np.array([1.0, -1.0]),
        )

        model.evaluation_mode = "sparse"

        expected = (
            scp.linalg.expm(1j * self.Z)
            @ ((1 + 1j) * self.Z + self.X)
            @ scp.linalg.expm(-1j * self.Z)
        )
        val = model(1.0)
        self.validate_generator_eval(val, expected)

    def validate_generator_eval(self, op, expected):
        """Validate that op is sparse and agrees with expected."""
        self.assertTrue(issparse(op))
        self.assertAllClose(unp.to_dense(op), unp.to_dense(expected))


class TestGeneratorModelSparseJax(TestGeneratorModelSparse):
    """JAX version of sparse model tests."""

    def validate_generator_eval(self, op, expected):
        """Validate that op is sparse and agrees with expected."""
        self.assertTrue(type(op).__name__ == "BCOO")
        self.assertAllClose(unp.to_dense(op), unp.to_dense(expected))

    def test_jit_grad(self):
        """Test jitting and gradding."""

        model = GeneratorModel(
            static_operator=-1j * self.Z,
            operators=[-1j * self.X],
            rotating_frame=self.Z,
            evaluation_mode="sparse",
        )

        y = np.array([0.0, 1.0])

        def func(a):
            model_copy = model.copy()
            model_copy.signals = [Signal(a)]
            return model_copy(0.232, y)

        jitted_func = self.jit_wrap(func)
        self.assertAllClose(jitted_func(1.2), func(1.2))

        grad_jit_func = self.jit_grad_wrap(func)
        grad_jit_func(1.2)


test_array_backends(TestGeneratorModelSparse, array_libraries=["numpy", "array_nunpy"])
test_array_backends(TestGeneratorModelSparseJax, array_libraries=["jax"])


class Testtransfer_operator_functions:
    """Tests for transfer_static_operator_between_frames and transfer_operators_between_frames."""

    def test_all_None(self):
        """Test all arguments being None."""

        static_operator = transfer_static_operator_between_frames(None, None, None)
        operators = transfer_operators_between_frames(None, None, None)

        self.assertTrue(static_operator is None)
        self.assertTrue(operators is None)

    def test_array_inputs_diagonal_frame(self):
        """Test correct handling when operators are arrays for diagonal frames."""

        static_operator = -1j * self.asarray([[1.0, 0.0], [0.0, -1.0]])
        operators = -1j * self.asarray([[[0.0, 1.0], [1.0, 0.0]], [[0.0, -1j], [1j, 0.0]]])
        new_frame = -1j * self.asarray([1.0, -1.0])

        out_static = transfer_static_operator_between_frames(static_operator, new_frame=new_frame)
        out_operators = transfer_operators_between_frames(operators, new_frame=new_frame)

        self.assertTrue(isinstance(out_static, ArrayLike))
        self.assertTrue(isinstance(out_operators, ArrayLike))

        self.assertAllClose(out_operators, operators)
        self.assertAllClose(out_static, np.zeros((2, 2)))

    def test_array_inputs_pseudo_random(self):
        """Test correct handling when operators are pseudo random arrays."""

        rng = np.random.default_rng(34233)
        num_terms = 3
        dim = 5
        b = 1.0  # bound on size of random terms
        static_operator = rng.uniform(low=-b, high=b, size=(dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(dim, dim)
        )
        operators = rng.uniform(low=-b, high=b, size=(num_terms, dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(dim, dim)
        )
        old_frame = rng.uniform(low=-b, high=b, size=(dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(dim, dim)
        )
        old_frame = RotatingFrame(old_frame - old_frame.conj().transpose())

        new_frame = rng.uniform(low=-b, high=b, size=(dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(dim, dim)
        )
        new_frame = RotatingFrame(new_frame - new_frame.conj().transpose())

        out_static = transfer_static_operator_between_frames(
            self.asarray(static_operator), new_frame=new_frame, old_frame=old_frame
        )
        out_operators = transfer_operators_between_frames(
            self.asarray(operators), new_frame=new_frame, old_frame=old_frame
        )

        self.assertTrue(isinstance(out_static, ArrayLike))
        self.assertTrue(isinstance(out_operators, ArrayLike))

        expected_static = new_frame.operator_into_frame_basis(
            (old_frame.operator_out_of_frame_basis(static_operator) + old_frame.frame_operator)
            - new_frame.frame_operator
        )
        expected_operators = [
            new_frame.operator_into_frame_basis(old_frame.operator_out_of_frame_basis(op))
            for op in operators
        ]

        self.assertAllClose(out_static, expected_static)
        self.assertAllClose(out_operators, expected_operators)


test_array_backends(
    Testtransfer_operator_functions, array_libraries=["numpy", "array_nunpy", "jax", "array_jax"]
)


class Testtransfer_operator_functionsSparse(QiskitDynamicsTestCase):
    """Tests for transfer_static_operator_between_frames and transfer_operators_between_frames
    for sparse cases.
    """

    def test_csr_inputs_diagonal_frame(self):
        """Test correct handling when operators are csr matrices with a diagonal frame."""

        static_operator = csr_matrix(-1j * np.array([[1.0, 0.0], [0.0, -1.0]]))
        operators = [
            -1j * csr_matrix([[0.0, 1.0], [1.0, 0.0]]),
            -1j * csr_matrix([[0.0, -1j], [1j, 0.0]]),
        ]
        new_frame = -1j * np.array([1.0, -1.0])

        out_static = transfer_static_operator_between_frames(static_operator, new_frame=new_frame)
        out_operators = transfer_operators_between_frames(operators, new_frame=new_frame)
        self.assertTrue(isinstance(out_static, csr_matrix))
        self.assertTrue(isinstance(out_operators, list) and issparse(out_operators[0]))

        self.assertAllCloseSparse(out_operators, operators)
        self.assertAllCloseSparse(out_static, csr_matrix(np.zeros((2, 2))))

    def test_csr_inputs_non_diagonal_frame(self):
        """Test correct handling when operators are csr matrices with non-diagonal frames."""

        static_operator = csr_matrix(-1j * np.array([[1.0, 0.0], [0.0, -1.0]]))
        operators = [
            -1j * csr_matrix([[0.0, 1.0], [1.0, 0.0]]),
            -1j * csr_matrix([[0.0, -1j], [1j, 0.0]]),
        ]
        old_frame = np.array(
            [
                [
                    0.0,
                    1.0,
                ],
                [1.0, 0.0],
            ]
        )
        new_frame = -1j * np.array([1.0, -1.0])

        _, U = np.linalg.eigh(old_frame)
        Uadj = U.conj().transpose()

        out_static = transfer_static_operator_between_frames(
            static_operator, new_frame=new_frame, old_frame=old_frame
        )
        out_operators = transfer_operators_between_frames(
            operators, new_frame=new_frame, old_frame=old_frame
        )

        self.assertTrue(isinstance(out_static, np.ndarray))
        self.assertTrue(
            isinstance(out_operators, list) and isinstance(out_operators[0], np.ndarray)
        )

        expected_ops = [U @ (op @ Uadj) for op in operators]
        expected_static = (
            U @ unp.to_dense(static_operator) @ Uadj + (-1j * old_frame) - np.diag(new_frame)
        )

        self.assertAllClose(out_operators, expected_ops)
        self.assertAllClose(out_static, expected_static)
