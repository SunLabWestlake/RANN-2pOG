from functools import lru_cache
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

from epg_analysis import COLORS, EXCITATION, PAIRS, POWER_MW, metrics_from_merged, single

ROOT = Path(__file__).resolve().parents[1]
FS = 12.0
TIME = (np.arange(90) - 19) / FS
DISTANCE = np.arange(8) * 22.5 + 11.25
ROI_PAIRS = np.array(list(PAIRS.values()))
DISTANCE_COLORS = ('#7f1727', '#b32632', '#d96150', '#efad9b',
                   '#f3dfd3', '#c7e0e5', '#84bbd4', '#2877ac')
RESULTS = {}


def configure():
    plt.rcParams.update({'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})


def finish(fig, panel):
    destination = ROOT / 'outputs' / 'analysis' / 'figures'
    destination.mkdir(parents=True, exist_ok=True)
    for extension in ('svg', 'pdf', 'png'):
        fig.savefig(destination / f'{panel}.{extension}', dpi=180, bbox_inches='tight')
    plt.show()


@lru_cache(None)
def repetitive_values(group, power=22, baseline_frames=15):
    code = next(code for code, mw in POWER_MW.items() if mw == power)
    path = ROOT / 'data' / 'repetitive' / f'repetitive_{group}_{code}.npz'
    with np.load(path, allow_pickle=False) as data:
        raw = data['raw']
        animal_ids = data['animal_ids'].astype(str)
    ids = np.array(list(dict.fromkeys(animal_ids)))
    corrected = raw - raw[:, :, :baseline_frames].mean(axis=2, keepdims=True)
    fly = np.stack([corrected[animal_ids == animal].mean(axis=0) for animal in ids])
    return fly, ids


def folded_values(group, power=22, baseline_frames=15):
    fly, ids = repetitive_values(group, power, baseline_frames)
    return fly[:, ROI_PAIRS].mean(axis=2), ids


def stimulation(ax, repeated=True):
    for onset in (np.arange(10) * 5 / FS if repeated else [0]):
        ax.axvspan(onset, onset + .05, ymin=0, ymax=.055,
                   color='#e57992', alpha=.8, lw=0)
    ax.set_xlabel('Time (s)')


def repetitive_traces(group, panel, power=22, baseline_frames=15):
    configure()
    values, _ = folded_values(group, power, baseline_frames)
    fig, ax = plt.subplots(figsize=(5.1, 2.7), layout='constrained')
    for distance, color, mean in zip(DISTANCE, DISTANCE_COLORS, values.mean(axis=0)):
        ax.plot(TIME, mean, color=color, lw=1.1, label=f'{distance:g}°')
    stimulation(ax)
    ax.set(ylabel=r'$\Delta F/F_0$', title=group, xlim=(TIME[0], TIME[-1]))
    ax.legend(fontsize=6, loc='upper left', bbox_to_anchor=(1.02, 1))
    finish(fig, panel)


def repetitive_raster(group, panel, power=22, baseline_frames=15):
    configure()
    values, _ = folded_values(group, power, baseline_frames)
    fig, ax = plt.subplots(figsize=(4.6, 2.7), layout='constrained')
    image = ax.imshow(values.mean(axis=0), origin='upper', aspect='auto',
        extent=(TIME[0] - .5 / FS, TIME[-1] + .5 / FS, 180, 0),
        interpolation='nearest', cmap='RdBu_r', vmin=-1, vmax=1)
    for onset in np.arange(10) * 5 / FS:
        ax.plot([onset, onset], [1.02, 1.09], transform=ax.get_xaxis_transform(),
                color='#c52b4b', lw=.8, clip_on=False)
    ax.set(xlabel='Time (s)', ylabel='Distance from stimulation (°)',
           yticks=[0, 45, 90, 135, 180], title=group)
    ax.set_title(group, pad=22)
    fig.colorbar(image, ax=ax, label=r'$\Delta F/F_0$')
    finish(fig, panel)


def cosine_values(group, power=22):
    """Fit b(t) + a(t) cos(d) beyond 90 degrees for each fly."""
    values, ids = folded_values(group, power, baseline_frames=10)
    far = DISTANCE >= 90
    design = np.column_stack((np.ones(far.sum()), np.cos(np.deg2rad(DISTANCE[far]))))
    coefficients = np.einsum('kd,ndt->nkt', np.linalg.pinv(design), values[:, far, :])
    return coefficients[:, 1], ids


def cosine_overlay(panel='Fig4M'):
    configure()
    fig, ax = plt.subplots(figsize=(4.0, 3.2), layout='constrained')
    for group in ('opChroME2s', 'opKCR1-C29D'):
        values, _ = cosine_values(group)
        mean, sem = values.mean(axis=0), stats.sem(values, axis=0)
        ax.plot(TIME, mean, color=COLORS[group], label=group, lw=1.5)
        ax.fill_between(TIME, mean - sem, mean + sem,
                        color=COLORS[group], alpha=.2, lw=0)
    ax.axhline(0, color='.7', lw=.6, ls=':')
    for time in (.5, 1, 2):
        ax.axvline(time, color='.7', lw=.6, ls=':')
    stimulation(ax)
    ax.set(ylabel='Cosine coefficient', xlim=(TIME[0], TIME[-1]), ylim=(-.75, .5))
    ax.legend(fontsize=7)
    finish(fig, panel)


def stimulated_repetitive(group, panel, power=10):
    configure()
    fly, _ = repetitive_values(group, power, baseline_frames=15)
    values = fly[:, 4, :]
    mean, sem = values.mean(axis=0), stats.sem(values, axis=0)
    fig, ax = plt.subplots(figsize=(4.2, 2.7), layout='constrained')
    ax.plot(TIME, values.T, color='.7', lw=.5, alpha=.7)
    ax.fill_between(TIME, mean - sem, mean + sem, color=COLORS[group], alpha=.2, lw=0)
    ax.plot(TIME, mean, color=COLORS[group], lw=1.4)
    stimulation(ax)
    ax.set(ylabel=r'$\Delta F/F_0$', title=group, xlim=(TIME[0], TIME[-1]))
    finish(fig, panel)


def antipodal_metric(metric, panel):
    configure()
    latency = metric == 'latency'
    key = ('antipodal_trough_time_s_frame19' if latency
           else 'antipodal_trough_dff_frames20_29')
    values = {group: metrics_from_merged(single(group)[0], group)[key]
              for group in EXCITATION}
    tests = []
    for a, b in combinations(EXCITATION, 2):
        test = stats.mannwhitneyu(values[a], values[b], alternative='two-sided', method='auto')
        tests.append(dict(group_a=a, group_b=b, U=test.statistic, p_raw=test.pvalue))
    for row, p in zip(tests, multipletests([row['p_raw'] for row in tests], method='holm')[1]):
        row['p_holm'] = p
    RESULTS[panel] = dict(fly_values=values, tests=tests)
    fig, ax = plt.subplots(figsize=(3.5, 3.2), layout='constrained')
    rng = np.random.default_rng(20261012)
    for i, group in enumerate(EXCITATION):
        y = values[group] * (1000 if latency else 100)
        ax.bar(i, y.mean(), yerr=stats.sem(y), color=COLORS[group],
               width=.55, capsize=3)
        ax.scatter(i + rng.uniform(-.06, .06, len(y)), y, s=20,
                   color='#535964', edgecolors='white', lw=.5, zorder=3)
    ax.set(xticks=range(3), xticklabels=EXCITATION,
           ylabel='Time to trough (ms)' if latency else r'Trough $\Delta F/F_0$ (%)')
    ax.tick_params(axis='x', rotation=30)
    finish(fig, panel)


def mean_normalized(values, negative=False):
    mean = values.mean(axis=0)
    scale = abs(mean.min() if negative else mean.max())
    if np.isclose(scale, 0):
        raise ValueError('Cannot normalize a zero group-mean response.')
    return values / scale


def antipodal_overlay(panel, normalized=False):
    configure()
    time = (np.arange(60) - 19) / FS
    fig, ax = plt.subplots(figsize=(4.7, 2.9), layout='constrained')
    values = {}
    for group in EXCITATION:
        y = single(group)[0][7].copy()
        if normalized:
            y = mean_normalized(y, negative=True)
        values[group] = y
        ax.plot(time, y.mean(axis=0), color=COLORS[group], label=group, lw=1.3)
    RESULTS[panel] = dict(fly_values=values)
    stimulation(ax, repeated=False)
    ax.set(ylabel='Normalized response' if normalized else r'$\Delta F/F_0$',
           xlim=(time[0], time[-1]))
    ax.legend(fontsize=7)
    finish(fig, panel)


def activation_inhibition(panel, comparison):
    configure()
    specs = {
        'stimulated': [('opChroME2s', 0, 1, False), ('opKCR1-C29D', 0, -1, True)],
        'antipodal': [('opChroME2s', 7, 1, True), ('opKCR1-C29D', 7, -1, False)],
        'cross_region': [('opChroME2s', 0, 1, False), ('opKCR1-C29D', 7, 1, False)],
    }
    time = (np.arange(60) - 19) / FS
    fig, ax = plt.subplots(figsize=(4.7, 2.9), layout='constrained')
    values = {}
    for group, region, sign, negative in specs[comparison]:
        y = sign * mean_normalized(single(group)[0][region], negative=negative)
        values[group] = y
        label = group + (' (sign-inverted)' if sign < 0 else '')
        ax.plot(time, y.mean(axis=0), color=COLORS[group], lw=1.3, label=label)
    RESULTS[panel] = dict(fly_values=values)
    stimulation(ax, repeated=False)
    ax.set(ylabel='Normalized response', xlim=(time[0], time[-1]))
    ax.legend(fontsize=7)
    finish(fig, panel)
