import numpy as np
import pytest
import torch.nn as nn

from simulation.core.integrator import (
    Dopri5,
    ExplicitEuler,
    ImplicitAdams,
    IntegratorBase,
    IntegratorFactory,
    IntegratorType,
    NeuralODE,
    VectorizedIntegratorBase,
)


class SimpleNeuralModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(6, 6)

    def forward(self, t, y):
        return self.net(y)


class TestIntegratorFactory:

    def test_create_basic_integrator(self):
        """Test creation of basic integrator types"""
        integrator = IntegratorFactory.create_integrator(IntegratorType.EXPLICIT_EULER.value)
        assert isinstance(integrator, ExplicitEuler)
        assert isinstance(integrator, VectorizedIntegratorBase)

    def test_unknown_integrator(self):
        """Test error handling for unknown integrator type"""
        with pytest.raises(ValueError, match="Unknown integrator: unknown_type"):
            IntegratorFactory.create_integrator("unknown_type")

    def test_neural_ode_creation(self):
        """Test creation of NeuralODE integrator"""
        model = SimpleNeuralModel()
        integrator = IntegratorFactory.create_integrator(
            IntegratorType.NEURAL_ODE.value, model=model
        )
        assert isinstance(integrator, NeuralODE)
        assert integrator.model == model

    def test_neural_ode_missing_model(self):
        """Test error handling when model is not provided for NeuralODE"""
        with pytest.raises(ValueError, match="NeuralODE requires a 'model' argument"):
            IntegratorFactory.create_integrator(IntegratorType.NEURAL_ODE.value)

    def test_implicit_adams_creation(self):
        """Test creation of ImplicitAdams integrator with custom order"""
        integrator = IntegratorFactory.create_integrator(
            IntegratorType.IMPLICIT_ADAMS.value, order=3
        )
        assert isinstance(integrator, ImplicitAdams)
        assert integrator.order == 3

    def test_dopri5_creation(self):
        """Test creation of Dopri5 integrator with custom tolerances"""
        integrator = IntegratorFactory.create_integrator(
            IntegratorType.DOPRI5.value, rtol=1e-4, atol=1e-7
        )
        assert isinstance(integrator, Dopri5)
        assert integrator.rtol == 1e-4
        assert integrator.atol == 1e-7

    def test_vectorized_integrator_n_jobs(self):
        """Test creation of vectorized integrator with custom n_jobs"""
        integrator = IntegratorFactory.create_integrator(
            IntegratorType.EXPLICIT_EULER.value, n_jobs=4
        )
        assert isinstance(integrator, VectorizedIntegratorBase)
        assert integrator.n_jobs == 4

    @pytest.mark.parametrize(
        "integrator_type",
        [
            IntegratorType.EXPLICIT_EULER.value,
            IntegratorType.SEMI_IMPLICIT_EULER.value,
            IntegratorType.IMPLICIT_EULER.value,
            IntegratorType.MIDPOINT.value,
            IntegratorType.RK2.value,
            IntegratorType.RK4.value,
            IntegratorType.RK38.value,
        ],
    )
    def test_all_basic_integrators(self, integrator_type):
        """Test creation of all basic integrator types"""
        integrator = IntegratorFactory.create_integrator(integrator_type)
        assert isinstance(integrator, IntegratorBase)
        assert isinstance(integrator, VectorizedIntegratorBase)

    def test_integrator_type_enum_values(self):
        """Test that all IntegratorType enum values are registered in factory"""
        for integrator_type in IntegratorType:
            assert integrator_type.value in IntegratorFactory._integrators

    def test_functional_integrator(self):
        """Test that created integrator is functional"""
        integrator = IntegratorFactory.create_integrator(IntegratorType.EXPLICIT_EULER.value)

        # Simple test case
        xt = np.array([0.0])
        vt = np.array([1.0])
        at = np.array([0.0])
        dt = 0.1

        xtp1, vtp1 = integrator.step(xt, vt, at, dt)
        assert isinstance(xtp1, np.ndarray)
        assert isinstance(vtp1, np.ndarray)
        assert xtp1.shape == xt.shape
        assert vtp1.shape == vt.shape

    @pytest.mark.parametrize(
        "integrator_type",
        [
            IntegratorType.EXPLICIT_EULER.value,
            IntegratorType.SEMI_IMPLICIT_EULER.value,
            IntegratorType.IMPLICIT_EULER.value,
            IntegratorType.MIDPOINT.value,
            IntegratorType.RK4.value,
        ],
    )
    def test_accuracy_against_analytical(self, integrator_type):
        """Test integrator accuracy against an analytical solution"""
        integrator = IntegratorFactory.create_integrator(integrator_type)

        # Analytical solution setup for SHO: x(t) = cos(t), v(t) = -sin(t)
        omega = 1.0  # Natural frequency
        xt = np.array([1.0])  # Initial position
        vt = np.array([0.0])  # Initial velocity
        at = np.array([-(omega**2) * xt[0]])  # Initial acceleration
        dt = 0.01  # Time step
        steps = 1000  # Number of steps

        # Numerical integration
        numerical_positions = [xt[0]]
        numerical_velocities = [vt[0]]

        for _ in range(steps):
            xt, vt = integrator.step(xt, vt, at, dt)
            at = np.array([-(omega**2) * xt[0]])  # Update acceleration
            numerical_positions.append(xt[0])
            numerical_velocities.append(vt[0])

        # Analytical solution
        times = np.arange(0, dt * (steps + 1), dt)
        analytical_positions = np.cos(omega * times)
        analytical_velocities = -omega * np.sin(omega * times)

        # Compute errors
        position_error = np.abs(np.array(numerical_positions) - analytical_positions)
        velocity_error = np.abs(np.array(numerical_velocities) - analytical_velocities)

        # Ensure errors are within acceptable bounds
        assert np.max(position_error) < 1e-2, f"Position error too large for {integrator_type}"
        assert np.max(velocity_error) < 1e-2, f"Velocity error too large for {integrator_type}"
