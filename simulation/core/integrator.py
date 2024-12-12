from enum import Enum
from multiprocessing import Pool

import numpy as np
import scipy.integrate as integrate
import torch
import torch.nn as nn
from numba import jit
from scipy.sparse import csc_matrix
from torchdiffeq import odeint


class IntegratorType(Enum):
    EXPLICIT_EULER = "explicit_euler"
    SEMI_IMPLICIT_EULER = "semi_implicit_euler"
    IMPLICIT_EULER = "implicit_euler"
    MIDPOINT = "midpoint"
    RK2 = "rk2"
    RK4 = "rk4"
    RK38 = "rk38"
    IMPLICIT_ADAMS = "implicit_adams"
    DOPRI5 = "dopri5"
    DOPRI3 = "dopri3"
    ADAPTIVE_HEUN = "adaptive_heun"
    VERLET = "verlet"
    LEAPFROG = "leapfrog"
    YOSHIDA = "yoshida"
    NUMBA_RK4 = "numba_rk4"
    SPARSE_EULER = "sparse_euler"
    DIRK = "dirk"
    HHT_ALPHA = "hht_alpha"
    ENERGY_MOMENTUM = "energy_momentum"
    SVK = "svk"
    NEURAL_ODE = "neural_ode"


class IntegratorBase:
    """Base class for integrators"""

    def step(self, xt: np.ndarray, vt: np.ndarray, at: np.ndarray, dt: float) -> tuple:
        raise NotImplementedError("step method not implemented")


class VectorizedIntegratorBase(IntegratorBase):
    """Base class for vectorized integrators"""

    def __init__(self, n_jobs: int = -1):
        self.n_jobs = n_jobs

    def _parallelize(self, func, data):
        if self.n_jobs != 1:
            with Pool(processes=self.n_jobs) as pool:
                return pool.map(func, data)
        return list(map(func, data))


# Basic Integrators
class ExplicitEuler(VectorizedIntegratorBase):
    def step(self, xt: np.ndarray, vt: np.ndarray, at: np.ndarray, dt: float) -> tuple:
        xtp1 = xt + dt * vt
        vtp1 = vt + dt * at
        return xtp1, vtp1


class SemiImplicitEuler(VectorizedIntegratorBase):
    def step(self, xt: np.ndarray, vt: np.ndarray, at: np.ndarray, dt: float) -> tuple:
        vtp1 = vt + dt * at
        xtp1 = xt + dt * np.multiply(vtp1, 1)
        return xtp1, vtp1


class ImplicitEuler(VectorizedIntegratorBase):
    def step(self, xt: np.ndarray, vt: np.ndarray, at: np.ndarray, dt: float) -> tuple:
        xtp1 = xt + dt * vt + 0.5 * dt * dt * at
        vtp1 = vt + dt * at
        return xtp1, vtp1


class Midpoint(VectorizedIntegratorBase):
    def step(self, xt: np.ndarray, vt: np.ndarray, at: np.ndarray, dt: float) -> tuple:
        mid_v = vt + 0.5 * dt * at
        mid_x = xt + 0.5 * dt * vt
        xtp1 = xt + dt * mid_v
        vtp1 = vt + dt * at
        return xtp1, vtp1


# Runge-Kutta Integrators
class RK2(VectorizedIntegratorBase):
    def step(self, xt: np.ndarray, vt: np.ndarray, at: np.ndarray, dt: float) -> tuple:
        k1x = vt
        k1v = at
        k2x = vt + 0.5 * dt * at
        k2v = at
        xtp1 = xt + dt * k2x
        vtp1 = vt + dt * k2v
        return xtp1, vtp1


class RK4(VectorizedIntegratorBase):
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


class ParallelRK4(VectorizedIntegratorBase):
    def _rk4_step(self, state):
        xt, vt, at, dt = state
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

    def step(self, xt: np.ndarray, vt: np.ndarray, at: np.ndarray, dt: float) -> tuple:
        states = [(x, v, a, dt) for x, v, a in zip(xt, vt, at)]
        results = self._parallelize(self._rk4_step, states)
        xtp1 = np.array([r[0] for r in results])
        vtp1 = np.array([r[1] for r in results])
        return xtp1, vtp1


class RK38(VectorizedIntegratorBase):
    def step(self, xt: np.ndarray, vt: np.ndarray, at: np.ndarray, dt: float) -> tuple:
        k1x = vt
        k1v = at
        k2x = vt + dt / 3 * at
        k2v = at
        k3x = vt - dt / 3 * at + dt * at
        k3v = at
        k4x = vt + dt * at - dt * at
        k4v = at

        xtp1 = xt + dt / 8 * (k1x + 3 * k2x + 3 * k3x + k4x)
        vtp1 = vt + dt / 8 * (k1v + 3 * k2v + 3 * k3v + k4v)
        return xtp1, vtp1


class ImplicitAdams(VectorizedIntegratorBase):
    def __init__(self, order: int = 3):
        self.order = order
        self.history_x = []
        self.history_v = []

    def step(self, xt: np.ndarray, vt: np.ndarray, at: np.ndarray, dt: float) -> tuple:
        self.history_x.append(xt)
        self.history_v.append(vt)

        if len(self.history_x) > self.order:
            self.history_x.pop(0)
            self.history_v.pop(0)

        if len(self.history_x) < self.order:
            rk4 = RK4()
            return rk4.step(xt, vt, at, dt)

        if self.order == 3:
            c = [23 / 12, -16 / 12, 5 / 12]
        else:
            raise ValueError(f"Order {self.order} not implemented")

        xtp1 = xt
        vtp1 = vt

        for i in range(self.order):
            xtp1 += dt * c[i] * self.history_v[-i - 1]
            vtp1 += dt * c[i] * at

        return xtp1, vtp1


class ScipyRK(VectorizedIntegratorBase):
    def __init__(self, method="RK45"):
        super().__init__()
        self.method = method

    def step(self, xt: np.ndarray, vt: np.ndarray, at: np.ndarray, dt: float) -> tuple:
        def system(t, y):
            x, v = np.split(y, 2)
            return np.concatenate([v, at])

        y0 = np.concatenate([xt, vt])
        sol = integrate.solve_ivp(system, [0, dt], y0, method=self.method, vectorized=True)

        xtp1, vtp1 = np.split(sol.y[:, -1], 2)
        return xtp1, vtp1


class Dopri5(ScipyRK):
    def __init__(self, rtol: float = 1e-3, atol: float = 1e-6):
        super().__init__(method="DOP853")
        self.rtol = rtol
        self.atol = atol


@jit(nopython=True)
def _numba_rk4_step(xt, vt, at, dt):
    k1x = vt
    k1v = at
    k2x = vt + 0.5 * dt * k1v
    k2v = at
    k3x = vt + 0.5 * dt * k2v
    k3v = at
    k4x = vt + dt * k3v
    k4v = at

    xtp1 = xt + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    vtp1 = vt + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return xtp1, vtp1


class NumbaRK4(VectorizedIntegratorBase):
    """JIT-compiled RK4 for maximum performance"""

    def step(self, xt, vt, at, dt):
        return _numba_rk4_step(xt, vt, at, dt)


class AdaptiveHeun(VectorizedIntegratorBase):
    """Adaptive step size Heun method - good balance of speed and accuracy"""

    def __init__(self, tol: float = 1e-6):
        super().__init__()
        self.tol = tol

    def step(self, xt: np.ndarray, vt: np.ndarray, at: np.ndarray, dt: float) -> tuple:
        # First stage
        k1x = vt
        k1v = at

        # Second stage
        xp = xt + dt * k1x
        vp = vt + dt * k1v
        k2x = vp
        k2v = at  # Evaluate acceleration at predicted point

        # Error estimate
        ex = dt / 2 * np.abs(k2x - k1x)
        ev = dt / 2 * np.abs(k2v - k1v)
        err = max(np.max(ex), np.max(ev))

        # Adapt step size
        if err > self.tol:
            dt *= 0.5
            return self.step(xt, vt, at, dt)

        xtp1 = xt + dt / 2 * (k1x + k2x)
        vtp1 = vt + dt / 2 * (k1v + k2v)
        return xtp1, vtp1


class Verlet(VectorizedIntegratorBase):
    """Velocity Verlet - excellent energy conservation"""

    def step(self, xt: np.ndarray, vt: np.ndarray, at: np.ndarray, dt: float) -> tuple:
        xtp1 = xt + vt * dt + 0.5 * at * dt**2
        vtp1 = vt + at * dt
        return xtp1, vtp1


class Leapfrog(VectorizedIntegratorBase):
    """Leapfrog integrator - symplectic, good for long-term stability"""

    def step(self, xt: np.ndarray, vt: np.ndarray, at: np.ndarray, dt: float) -> tuple:
        vthalf = vt + 0.5 * dt * at
        xtp1 = xt + dt * vthalf
        vtp1 = vthalf + 0.5 * dt * at
        return xtp1, vtp1


class Yoshida(VectorizedIntegratorBase):
    """4th order symplectic integrator"""

    def __init__(self):
        super().__init__()
        self.w0 = -(2 ** (1 / 3)) / (2 - 2 ** (1 / 3))
        self.w1 = 1 / (2 - 2 ** (1 / 3))

    def step(self, xt: np.ndarray, vt: np.ndarray, at: np.ndarray, dt: float) -> tuple:
        # Coefficients for 4th order symplectic integration
        c = [self.w1 / 2, (self.w0 + self.w1) / 2, (self.w0 + self.w1) / 2, self.w1 / 2]
        d = [self.w1, self.w0, self.w1]

        x, v = xt.copy(), vt.copy()

        for i in range(4):
            x += c[i] * dt * v
            if i < 3:
                v += d[i] * dt * at

        return x, v


class SparseEuler(VectorizedIntegratorBase):
    """Euler method optimized for sparse systems"""

    def step(self, xt: np.ndarray, vt: np.ndarray, at: np.ndarray, dt: float) -> tuple:
        # Convert to sparse format if dense
        if not isinstance(at, csc_matrix):
            at = csc_matrix(at)

        xtp1 = xt + dt * vt
        vtp1 = vt + dt * at.dot(xt)  # Efficient sparse matrix multiplication
        return xtp1, vtp1


class DIRKIntegrator(VectorizedIntegratorBase):
    """Diagonally Implicit Runge-Kutta for stiff systems"""

    def __init__(self, gamma: float = 0.435866521508459):
        super().__init__()
        self.gamma = gamma  # Optimal value for L-stability

    def step(self, xt: np.ndarray, vt: np.ndarray, at: np.ndarray, dt: float) -> tuple:
        # Single-stage DIRK (aka implicit midpoint)
        k1 = at

        # Implicit stage solution (simplified Newton)
        for _ in range(3):  # Fixed number of iterations
            k_new = at + self.gamma * dt * k1
            if np.allclose(k1, k_new):
                break
            k1 = k_new

        xtp1 = xt + dt * vt + 0.5 * dt * dt * k1
        vtp1 = vt + dt * k1
        return xtp1, vtp1


class HHTAlpha(VectorizedIntegratorBase):
    """HHT-α method for structural dynamics"""

    def __init__(self, alpha: float = -0.3):
        super().__init__()
        self.alpha = alpha
        self.gamma = 0.5 - alpha
        self.beta = 0.25 * (1 - alpha) ** 2

    def step(self, xt: np.ndarray, vt: np.ndarray, at: np.ndarray, dt: float) -> tuple:
        # Predict
        xtp1 = xt + dt * vt + dt**2 * ((0.5 - self.beta) * at)
        vtp1 = vt + dt * ((1 - self.gamma) * at)

        # Correct (single iteration for demonstration)
        atp1 = (1 - self.alpha) * at  # Simplified correction

        # Update
        xtp1 += dt**2 * self.beta * atp1
        vtp1 += dt * self.gamma * atp1
        return xtp1, vtp1


class EnergyMomentum(VectorizedIntegratorBase):
    """Energy-Momentum conserving integrator for hyperelasticity"""

    def step(self, xt: np.ndarray, vt: np.ndarray, at: np.ndarray, dt: float) -> tuple:
        # Midpoint configuration
        x_mid = 0.5 * (xt + xt + dt * vt)
        v_mid = 0.5 * (vt + vt + dt * at)

        # Discrete gradient approximation
        xtp1 = xt + dt * v_mid
        vtp1 = vt + dt * at

        # Energy correction
        kinetic_energy = 0.5 * np.sum(vtp1**2)
        initial_energy = 0.5 * np.sum(vt**2)
        scale = np.sqrt(initial_energy / (kinetic_energy + 1e-10))
        vtp1 *= scale

        return xtp1, vtp1


class SVKIntegrator(VectorizedIntegratorBase):
    """Specialized integrator for St. Venant-Kirchhoff materials"""

    def __init__(self, young: float, poisson: float):
        super().__init__()
        self.lmbda = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
        self.mu = young / (2 * (1 + poisson))

    def step(self, xt: np.ndarray, vt: np.ndarray, at: np.ndarray, dt: float) -> tuple:
        # Compute deformation gradient (simplified)
        F = np.eye(3) + np.gradient(xt, axis=0)

        # Green strain
        E = 0.5 * (F.T @ F - np.eye(3))

        # PK2 stress
        S = self.lmbda * np.trace(E) * np.eye(3) + 2 * self.mu * E

        # Elastic force (simplified)
        f_elastic = -np.einsum("ij,jk->ik", F, S).ravel()

        # Integration
        xtp1 = xt + dt * vt
        vtp1 = vt + dt * (at + f_elastic)
        return xtp1, vtp1


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

        xtp1, vtp1 = (
            final_state[..., : xt.shape[-1]],
            final_state[..., xt.shape[-1] : 2 * xt.shape[-1]],
        )
        return xtp1, vtp1


class IntegratorFactory:
    _integrators = {
        IntegratorType.EXPLICIT_EULER.value: ExplicitEuler,
        IntegratorType.SEMI_IMPLICIT_EULER.value: SemiImplicitEuler,
        IntegratorType.IMPLICIT_EULER.value: ImplicitEuler,
        IntegratorType.MIDPOINT.value: Midpoint,
        IntegratorType.RK2.value: ScipyRK,
        IntegratorType.RK4.value: ParallelRK4,
        IntegratorType.RK38.value: RK38,
        IntegratorType.IMPLICIT_ADAMS.value: ImplicitAdams,
        IntegratorType.DOPRI5.value: Dopri5,
        IntegratorType.ADAPTIVE_HEUN.value: AdaptiveHeun,
        IntegratorType.VERLET.value: Verlet,
        IntegratorType.LEAPFROG.value: Leapfrog,
        IntegratorType.YOSHIDA.value: Yoshida,
        IntegratorType.NUMBA_RK4.value: NumbaRK4,
        IntegratorType.SPARSE_EULER.value: SparseEuler,
        IntegratorType.DIRK.value: DIRKIntegrator,
        IntegratorType.HHT_ALPHA.value: HHTAlpha,
        IntegratorType.ENERGY_MOMENTUM.value: EnergyMomentum,
        IntegratorType.SVK.value: SVKIntegrator,
        IntegratorType.NEURAL_ODE.value: NeuralODE,
    }

    @staticmethod
    def create_integrator(name: str, **kwargs) -> IntegratorBase:
        if name not in IntegratorFactory._integrators:
            raise ValueError(f"Unknown integrator: {name}")

        n_jobs = kwargs.pop("n_jobs", -1)

        integrator_cls = IntegratorFactory._integrators[name]

        if name == IntegratorType.NEURAL_ODE.value:
            if "model" not in kwargs:
                raise ValueError("NeuralODE requires a 'model' argument")
            return integrator_cls(kwargs["model"])

        if name == IntegratorType.IMPLICIT_ADAMS.value:
            order = kwargs.get("order", 3)
            return integrator_cls(order=order)

        if name == IntegratorType.DOPRI5.value:
            rtol = kwargs.get("rtol", 1e-3)
            atol = kwargs.get("atol", 1e-6)
            return integrator_cls(rtol=rtol, atol=atol)

        if issubclass(integrator_cls, VectorizedIntegratorBase):
            return integrator_cls(n_jobs=n_jobs)

        return integrator_cls()
