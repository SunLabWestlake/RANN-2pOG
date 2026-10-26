"""Uniform and connectome-constrained EPG–Δ7 dynamics."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from .ring_model import Ring2019Params

DATA = Path(__file__).resolve().parents[1] / 'data' / 'model' / 'connectomes'


@lru_cache(None)
def system(dataset='hemibrain', fraction=0.5, tau_delta_s=0.02, kappa=-0.5):
    """Return postsynaptic-by-presynaptic weights and settled initial states."""
    p = Ring2019Params()
    if dataset == 'uniform':
        return (p.weight_matrix(), np.ones(32), np.full(32, p.tau_s),
                np.stack([p.stationary_bump(j) for j in range(32)]))
    with np.load(DATA / f'{dataset}.npz', allow_pickle=False) as saved:
        A, B, C, states = [saved[k].copy() for k in ('A', 'B', 'C', 'states')]
    drive = A @ np.ones(32)

    def equilibrium_rhs(t, delta):
        return np.clip(drive - kappa * C @ delta, 0, 1000) - delta

    solved = solve_ivp(equilibrium_rhs, (0, 80), np.maximum(drive, 0),
                       rtol=1e-9, atol=1e-11)
    if not solved.success:
        raise RuntimeError(solved.message)
    gain = 32 * p.beta_discrete / np.mean(B @ solved.y[:, -1])
    n_delta = len(A)
    weights = np.zeros((32 + n_delta, 32 + n_delta))
    weights[:32, :32] = p.weight_matrix() + fraction * p.beta_discrete
    weights[:32, 32:] = -fraction * gain * B
    weights[32:, :32] = A
    weights[32:, 32:] = -kappa * C
    bias = np.r_[np.ones(32), np.zeros(n_delta)]
    tau = np.r_[np.full(32, p.tau_s), np.full(n_delta, tau_delta_s)]
    return weights, bias, tau, states


def _interpolate(values, time):
    index = min(max(int(time / .01), 0), values.shape[-1] - 2)
    fraction = (time - index * .01) / .01
    return values[..., index] + fraction * (values[..., index + 1] - values[..., index])


def integrate(starts, weights, bias, tau, velocities, spatials, temporal,
              n_frames, substeps=40):
    """Fourth-order Runge–Kutta, sampled at 12 Hz with 40 substeps per frame."""
    values = starts.copy()
    result = np.empty((*values.shape, n_frames))
    result[:, :, 0] = values
    step_size = 1 / (12 * substeps)

    def derivative(state, velocity, stimulus):
        drive = state @ weights.T + bias
        epg = state[:, :32]
        motion = (np.roll(epg, 1, axis=1) - np.roll(epg, -1, axis=1)) / 2
        drive[:, :32] += velocity[:, None] * 32 / (2 * np.pi) * motion
        drive[:, :32] += spatials * stimulus * 2 * np.pi / 32
        return (np.clip(drive, 0, 1000) - state) / tau

    for frame in range(1, n_frames):
        for step in range(substeps):
            time = ((frame - 1) * substeps + step) * step_size
            v0, vm, v1 = [_interpolate(velocities, time + f * step_size) for f in (0, .5, 1)]
            u0, um, u1 = [_interpolate(temporal, time + f * step_size) for f in (0, .5, 1)]
            a = derivative(values, v0, u0)
            b = derivative(values + step_size * .5 * a, vm, um)
            c = derivative(values + step_size * .5 * b, vm, um)
            d = derivative(values + step_size * c, v1, u1)
            values += step_size * (a + 2 * b + 2 * c + d) / 6
        result[:, :, frame] = values
    if not np.isfinite(result).all() or result.min() < -1e-7:
        raise FloatingPointError('Invalid simulated activity')
    return result
