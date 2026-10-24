"""Ring parameters and opsin input kernels."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Sequence

import numpy as np

from scipy.ndimage import gaussian_filter1d

TWO_PI = 2.0 * np.pi

@dataclass(frozen=True)
class Ring2019Params:
    """Parameters for the EPG ring-attractor model."""
    n_wedge: int = 32
    tau_s: float = 0.08
    bump_width: float = 0.5 * np.pi * 0.8
    D_cont: float = 0.3
    beta_cont: float = 5.0
    membrane_threshold: float = 0
    membrane_saturation: float = 1000.0

    def __post_init__(self) -> None:
        m = self.n_wedge / TWO_PI * self.bump_width
        c2d = TWO_PI / self.n_wedge
        D = self.D_cont / c2d ** 2
        alpha_ = (np.sin(np.pi / (m - 0.5 + 2.0)) * 2.0) ** 2 * D + 1.0
        beta_discrete = self.beta_cont * c2d
        omega = np.arcsin(np.sqrt((alpha_ - 1.0) / D) / 2.0) * 2.0
        M = int(np.ceil(2.0 * np.pi / omega - 2.0))
        phi = np.arctan(np.sin((M + 1) * omega) / (np.cos((M + 1) * omega) - 1.0))
        tmp = (1.0 - alpha_) * np.sin(phi) + beta_discrete * (np.sin(omega) * np.cos(phi) / (1.0 - np.cos(omega)) + (M + 1) * np.sin(phi))
        A = 1.0 / tmp
        S = A * (np.sin(omega) * np.cos(phi) / (1.0 - np.cos(omega)) + (M + 1) * np.sin(phi))
        beta_min = (alpha_ - 1.0) * np.sin(phi) / (np.sin(omega) * np.cos(phi) / (1.0 - np.cos(omega)) + (M + 1) * np.sin(phi))
        if self.beta_cont < beta_min / c2d:
            raise ValueError(f'beta_cont must be at least {beta_min / c2d:.4f}.')
        for key, value in (('alpha_', alpha_), ('D', D), ('beta_discrete', beta_discrete), ('omega', omega), ('M', M), ('phi', phi), ('A', A), ('S', S)):
            object.__setattr__(self, key, value)

    def weight_matrix(self) -> np.ndarray:
        """W_ring_attractor: alpha-2D-beta diagonal, D-beta neighbors, -beta else."""
        n = self.n_wedge
        W = np.full((n, n), -self.beta_discrete)
        idx = np.arange(n)
        W[idx, idx] = self.alpha_ - 2.0 * self.D - self.beta_discrete
        W[idx, (idx - 1) % n] = self.D - self.beta_discrete
        W[idx, (idx + 1) % n] = self.D - self.beta_discrete
        return W

    def stationary_bump(self, center_wedge: int) -> np.ndarray:
        """Periodic analytic stationary bump centered on one of 32 wedges."""
        center_wedge = int(center_wedge) % self.n_wedge
        if self.M % 2 != 0:
            raise ValueError('This initializer currently requires an even M.')
        j = np.arange(self.M + 1, dtype=float)
        indices = (center_wedge - self.M // 2 + np.arange(self.M + 1)) % self.n_wedge
        bump = np.zeros(self.n_wedge)
        bump[indices] = self.A * (np.sin(self.omega * j - self.phi) + np.sin(self.phi))
        return bump

def extract_and_interpolate_trace(trace: Sequence[float], stim_onset: int=19, *, sample_rate_hz: float=12.0, dt: float=0.01, legacy_stretch: bool=False) -> tuple[np.ndarray, np.ndarray]:
    """(t_support, values) of the causal kernel sampled at 10 ms."""
    values = np.asarray(trace, dtype=float)
    causal = values[stim_onset:] - float(values[stim_onset])
    t_support = np.arange(len(causal), dtype=float) / sample_rate_hz
    if legacy_stretch:
        t_out = np.arange(500, dtype=float) * dt
    else:
        t_out = np.arange(0.0, t_support[-1] + dt, dt)
    return (t_out, np.interp(t_out, t_support, causal, left=0.0, right=causal[-1]))

def plateau_wedge_profile(params: Ring2019Params, center_wedge: float, kappa: float=10.0) -> np.ndarray:
    """Official optogenetic spatial profile (sim_cond.m), peak-normalized to 1."""
    theta = np.arange(params.n_wedge) * TWO_PI / params.n_wedge
    mu = float(center_wedge) * TWO_PI / params.n_wedge
    profile = np.exp(kappa * (np.cos(theta - mu) - 1.0))
    profile = np.where(profile > profile.max() / 2.0, profile.max(), profile)
    return (profile - profile.min()) / (profile.max() - profile.min())

def flat_wedge_profile(params: Ring2019Params, first_wedge: int, n_wedges: int=4) -> np.ndarray:
    """Hard rectangular site: two 16-ROI sectors correspond to four model nodes."""
    profile = np.zeros(params.n_wedge)
    profile[(first_wedge + np.arange(n_wedges)) % params.n_wedge] = 1.0
    return profile

def site_spatial_profile(params: Ring2019Params, site_rois: Sequence[int], profile: str='plateau') -> np.ndarray:
    """Map a 16-ROI site to 32-wedge nodes; site (3,4) -> wedges 6-9."""
    rois = tuple((int(i) % 16 for i in site_rois))
    first_wedge = min(rois) * 2
    n_wedges = 2 * len(rois)
    if profile == 'flat':
        return flat_wedge_profile(params, first_wedge, n_wedges)
    if profile == 'plateau':
        center_wedge = first_wedge + 0.5 * (n_wedges - 1)
        return plateau_wedge_profile(params, center_wedge)
    raise ValueError("profile must be 'plateau' or 'flat'.")

def smooth_velocity_noise(t_max: float, *, dt: float=0.01, sigma: float=20.0, vel_amp: float=0.01, seed: int | None=None) -> tuple[np.ndarray, np.ndarray]:
    """Study-specific smoothed turning drive. Its amplitude is not calibrated to physical angular speed; actual bump motion depends on network parameters."""
    if vel_amp == 0:
        return (np.array([0.0, t_max]), np.zeros(2))
    rng = np.random.default_rng(seed)
    t_steps = np.arange(0.0, t_max + dt, dt)
    raw = rng.standard_normal(len(t_steps))
    smooth = gaussian_filter1d(raw, sigma=sigma)
    smooth = smooth / np.max(np.abs(smooth)) * vel_amp
    return (t_steps, smooth)

def build_stimulus_array(params: Ring2019Params, events: Sequence[tuple[float, float, np.ndarray, np.ndarray, float | None]], t_max: float, dt: float=0.01) -> np.ndarray:
    """(n_wedge, n_steps) injection array; events = (onset_s, gain, kernel, profile, truncate_s)."""
    n_steps = int(round(t_max / dt)) + 1
    stim = np.zeros((params.n_wedge, n_steps))
    for onset, gain, kernel, profile, truncate in events:
        t0 = int(round(onset / dt))
        n_kernel = len(kernel) if truncate is None else min(len(kernel), int(round(truncate / dt)))
        n_use = min(n_kernel, n_steps - t0)
        if n_use > 0 and t0 < n_steps:
            stim[:, t0:t0 + n_use] += float(gain) * np.outer(profile, kernel[:n_use])
    return stim

def preprocess_opsin_kernel(kernel, polarity, *, dt=0.01, smooth_sigma_s=0.1, end_taper_s=0.3):
    """
    polarity:
        "excitatory" -> output range [0, 1]
        "inhibitory" -> output range [-1, 0]

    The number of time samples and dt are unchanged.
    """
    kernel = np.asarray(kernel, dtype=float).copy()
    if kernel.ndim != 1:
        raise ValueError('kernel must be one-dimensional.')
    if polarity == 'excitatory':
        magnitude = np.clip(kernel, 0.0, None)
    elif polarity == 'inhibitory':
        magnitude = np.clip(-kernel, 0.0, None)
    else:
        raise ValueError("polarity must be 'excitatory' or 'inhibitory'.")
    sigma_samples = smooth_sigma_s / dt
    if sigma_samples > 0:
        magnitude = gaussian_filter1d(magnitude, sigma=sigma_samples, mode='nearest')
    magnitude[0] = 0.0
    if end_taper_s is not None and end_taper_s > 0:
        n_taper = int(round(end_taper_s / dt))
        n_taper = min(n_taper, len(magnitude))
        if n_taper > 1:
            taper = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, n_taper)))
            magnitude[-n_taper:] *= taper
    magnitude = np.clip(magnitude, 0.0, None)
    peak = magnitude.max()
    if peak <= 0:
        raise ValueError('Kernel contains no signal after polarity constraint.')
    magnitude /= peak
    if polarity == 'excitatory':
        output = magnitude
    else:
        output = -magnitude
    output[0] = 0.0
    output[-1] = 0.0
    return output
