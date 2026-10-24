"""Prepared model inputs, reproducible simulations, and panel readouts."""
from pathlib import Path
import numpy as np
from . import ring_model as model
from .dynamics import system, integrate

DATA = Path(__file__).resolve().parents[1] / 'data' / 'model'
OPSINS = ('ChRmine', 'ChroME2s', 'CsChrimson', 'KCR')
LABELS = dict(zip(OPSINS, ('opChRmine', 'opChroME2s', 'CsChrimson', 'opKCR1-C29D')))
DATASETS = ('uniform', 'hemibrain', 'banc', 'malecns')
DISTANCE_PAIRS = np.array([[3, 4], [2, 5], [1, 6], [0, 7],
                           [15, 8], [14, 9], [13, 10], [12, 11]])
DISTANCE_PI = np.arange(8) / 8
GAINS = {'ChRmine': (8., 18.73, 25., 18.73),
         'ChroME2s': (8., 12.86, 19.9, 12.86),
         'CsChrimson': (7.9, 8.9, 10.07, 8.9),
         'KCR': (1., 2., 3., 8.)}


def specification(opsin, condition='mid'):
    repeated = condition == 'repetitive'
    return dict(gain=GAINS[opsin][('low', 'mid', 'high', 'repetitive').index(condition)],
                n_runs=16 if repeated else (500 if opsin == 'KCR' else 100),
                n_frames=90 if repeated else 60, base_seed=20260817, vel_amp=.2,
                onset_s=20 / 12 if repeated else 1.6)


def empirical_traces():
    with np.load(DATA / 'empirical_traces.npz', allow_pickle=False) as saved:
        return {name: saved[name].copy() for name in OPSINS}


def load_kernels():
    result = {}
    for name, raw in empirical_traces().items():
        _, waveform = model.extract_and_interpolate_trace(raw, stim_onset=19)
        result[name] = model.preprocess_opsin_kernel(
            waveform, 'inhibitory' if name == 'KCR' else 'excitatory')
    return result


def load_result(opsin, condition='mid', *, dataset='uniform'):
    with np.load(DATA / 'cached' / f'{dataset}__{opsin}_{condition}.npz',
                 allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def distance_activity(activity):
    values = np.asarray(activity, dtype=float)
    delta = values - values[:, :, :20].mean(axis=2, keepdims=True)
    return delta[:, DISTANCE_PAIRS, :].mean(axis=2)


def summary(activity):
    values = distance_activity(activity)
    sem = values.std(axis=0, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.zeros_like(values[0])
    return values.mean(axis=0), sem


def normalized_spatial_auc(activity):
    auc = (distance_activity(activity)[:, :, 20:40].sum(axis=2) / 12).mean(axis=0)
    span = np.ptp(auc)
    if span <= 0:
        raise ValueError('Cannot normalize a constant spatial profile')
    return auc, (auc - auc.min()) / span


def simulate(opsin, condition='mid', *, dataset='uniform', n_runs=None, seed=20260817):
    """Recompute a complete condition in memory; no files are written."""
    spec = specification(opsin, condition)
    n = spec['n_runs'] if n_runs is None else n_runs
    seeds = np.array([s.generate_state(1, dtype=np.uint32)[0]
                      for s in np.random.SeedSequence(seed).spawn(n)], dtype=np.uint32)
    order = np.random.default_rng(np.random.SeedSequence([seed, 617])).permutation(16)
    rotations = np.zeros(n, dtype=int) if dataset == 'uniform' else np.resize(order, n)
    p = model.Ring2019Params()
    spatial = model.site_spatial_profile(p, (3, 4), 'plateau')
    kernel = load_kernels()[opsin]
    events = ([(f / 12, spec['gain'], kernel, spatial, 1.5) for f in range(20, 70, 5)]
              if condition == 'repetitive' else [(1.6, spec['gain'], kernel, spatial, None)])
    duration = (spec['n_frames'] - 1) / 12
    stimulus = model.build_stimulus_array(p, events, duration)
    anchor = np.argmax(spatial)
    temporal = stimulus[anchor] / spatial[anchor]
    centers, velocities, spatials = [], [], []
    for random_seed, rotation in zip(seeds, rotations):
        initial_seed, velocity_seed = np.random.SeedSequence(int(random_seed)).spawn(2)
        centers.append(int(np.random.default_rng(initial_seed).integers(0, 32)))
        velocity_seed = int(velocity_seed.generate_state(1, dtype=np.uint32)[0])
        velocities.append(model.smooth_velocity_noise(duration, vel_amp=spec['vel_amp'], seed=velocity_seed)[1])
        spatials.append(np.roll(spatial, 2 * int(rotation)))
    centers = np.asarray(centers)
    weights, bias, tau, states = system(dataset)
    starts = states[(centers + 2 * rotations) % 32]
    activity = integrate(starts, weights, bias, tau, np.asarray(velocities),
                         np.asarray(spatials), temporal, spec['n_frames'])
    aligned = np.stack([np.roll(trial[:32], -2 * int(rotation), axis=0)
                        for trial, rotation in zip(activity, rotations)])
    return dict(activity_16=aligned.reshape(n, 16, 2, -1).mean(axis=2),
                times_s=np.arange(spec['n_frames']) / 12, seeds=seeds,
                initial_center_wedge=centers, rotation_roi=rotations)
