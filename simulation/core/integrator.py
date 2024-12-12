from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint

# Base class for integrators
class IntegratorBase(ABC):
    @abstractmethod
    def step(self, xt: np.ndarray, vt: np.ndarray, at: np.ndarray, dt: float) -> tuple:
        """Perform one integration step"""
        pass

# Explicit Euler integrator
class ExplicitEuler(IntegratorBase):
    def step(self, xt: np.ndarray, vt: np.ndarray, at: np.ndarray, dt: float) -> tuple:
        xtp1 = xt + dt * vt
        vtp1 = vt + dt * at
        return xtp1, vtp1

# Semi-implicit Euler integrator
class SemiImplicitEuler(IntegratorBase):
    def step(self, xt: np.ndarray, vt: np.ndarray, at: np.ndarray, dt: float) -> tuple:
        vtp1 = vt + dt * at
        xtp1 = xt + dt * vtp1
        return xtp1, vtp1

# Implicit Euler integrator
class ImplicitEuler(IntegratorBase):
    def step(self, xt: np.ndarray, vt: np.ndarray, at: np.ndarray, dt: float) -> tuple:
        xtp1 = xt + dt * vt + 0.5 * dt * dt * at
        vtp1 = vt + dt * at
        return xtp1, vtp1

# Midpoint integrator
class Midpoint(IntegratorBase):
    def step(self, xt: np.ndarray, vt: np.ndarray, at: np.ndarray, dt: float) -> tuple:
        mid_v = vt + 0.5 * dt * at
        mid_x = xt + 0.5 * dt * vt
        xtp1 = xt + dt * mid_v
        vtp1 = vt + dt * at
        return xtp1, vtp1

# Runge-Kutta 2nd Order integrator
class RK2(IntegratorBase):
    def step(self, xt: np.ndarray, vt: np.ndarray, at: np.ndarray, dt: float) -> tuple:
        k1x = vt
        k1v = at

        k2x = vt + 0.5 * dt * at
        k2v = at  # Assuming constant acceleration in this example

        xtp1 = xt + dt * k2x
        vtp1 = vt + dt * k2v
        return xtp1, vtp1

# Runge-Kutta 4th Order integrator
class RK4(IntegratorBase):
    def step(self, xt: np.ndarray, vt: np.ndarray, at: np.ndarray, dt: float) -> tuple:
        k1x = vt
        k2x = vt + 0.5 * dt * at
        k3x = vt + 0.5 * dt * at
        k4x = vt + dt * at

        k1v = at
        k2v = at
        k3v = at
        k4v = at

        xtp1 = xt + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        vtp1 = vt + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        
        return xtp1, vtp1

# NeuralODE class
class NeuralODE(IntegratorBase):
    def __init__(self, model: nn.Module):
        self.model = model

    def step(self, xt: np.ndarray, vt: np.ndarray, at: np.ndarray, dt: float) -> tuple:
        xt_torch = torch.tensor(xt, dtype=torch.float32)
        vt_torch = torch.tensor(vt, dtype=torch.float32)
        at_torch = torch.tensor(at, dtype=torch.float32)

        initial_state = torch.cat([xt_torch, vt_torch, at_torch], dim=-1)
        t_span = torch.tensor([0, dt], dtype=torch.float32)

        solution = odeint(self.model, initial_state, t_span)
        final_state = solution[-1].detach().numpy()

        xtp1, vtp1 = final_state[..., :xt.shape[-1]], final_state[..., xt.shape[-1]:2*xt.shape[-1]]
        return xtp1, vtp1

# Neural ODE Model
class ODEModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(ODEModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, t, y):
        return self.net(y)

# Integrator Factory
class IntegratorFactory:
    _integrators = {
        "explicit_euler": ExplicitEuler,
        "semi_implicit_euler": SemiImplicitEuler,
        "implicit_euler": ImplicitEuler,
        "midpoint": Midpoint,
        "rk2": RK2,
        "rk4": RK4,
        "neural_ode": NeuralODE
    }

    @staticmethod
    def create_integrator(name: str, **kwargs) -> IntegratorBase:
        if name not in IntegratorFactory._integrators:
            raise ValueError(f"Unknown integrator: {name}")

        if name == "neural_ode":
            if "model" not in kwargs:
                raise ValueError("NeuralODE requires a 'model' argument")
            return IntegratorFactory._integrators[name](kwargs["model"])

        return IntegratorFactory._integrators[name]()