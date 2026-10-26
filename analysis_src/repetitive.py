from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.integrate import trapezoid
from statsmodels.stats.multitest import multipletests

from epg_analysis import EXCITATION, COLORS
from epg_panels import save, mean_bar_with_fly_points
from spatial_panels import configure, repetitive_values

RESULTS = {}


@lru_cache(None)
def spectra():
    result = {}
    for group in EXCITATION:
        fly, ids = repetitive_values(group, power=22, baseline_frames=15)
        values = signal.detrend(fly[:, 4, :], axis=-1, type='linear')
        frequency, power = signal.welch(values, fs=12, window='hann', nperseg=36,
            noverlap=18, nfft=36, detrend=False, scaling='density', axis=-1)
        band = (frequency >= .5) & (frequency <= 6)
        f, p = frequency[band], power[:, band]
        integral = trapezoid(p, x=f, axis=1)
        if np.any(integral <= 0):
            raise ValueError('Non-positive integrated PSD')
        result[group] = dict(frequency=f, db=10*np.log10(np.maximum(p, np.finfo(float).tiny)),
                             relative=p/integral[:, None], fly_ids=ids)
    return result


def psd(panel='Fig2N'):
    configure()
    fig, ax = plt.subplots(figsize=(4.2, 2.8), layout='constrained')
    for group, data in spectra().items():
        mean, sem = data['db'].mean(axis=0), stats.sem(data['db'], axis=0)
        f = data['frequency']
        ax.plot(f, mean, color=COLORS[group], label=group)
        ax.fill_between(f, mean-sem, mean+sem, color=COLORS[group], alpha=.15, lw=0)
    ax.set(xlabel='Frequency (Hz)', ylabel='PSD (dB)', xlim=(.5, 6))
    ax.legend(fontsize=7)
    save(fig, panel)
    plt.show()


def relative_psd(panel='Fig2O'):
    """Sum normalized spectral-density bins at 5/3, 2 and 7/3 Hz per fly."""
    configure()
    values = {group: data['relative'][:, 3:6].sum(axis=1)
              for group, data in spectra().items()}
    tests = []
    for group in EXCITATION[1:]:
        test = stats.mannwhitneyu(values[group], values['CsChrimson'],
                                 alternative='two-sided', method='asymptotic', use_continuity=True)
        tests.append(dict(group_a=group, group_b='CsChrimson', U=test.statistic, p_raw=test.pvalue))
    for row, p in zip(tests, multipletests([row['p_raw'] for row in tests], method='holm')[1]):
        row['p_holm'] = p
    RESULTS[panel] = dict(fly_values=values, tests=tests)
    fig, ax = plt.subplots(figsize=(4.2, 2.8), layout='constrained')
    rng = np.random.default_rng(20261013)
    order = list(reversed(EXCITATION))
    for i, group in enumerate(order):
        y = values[group]
        mean_bar_with_fly_points(ax, i, y, COLORS[group], rng.uniform(-.12, .12, len(y)))
    ax.set(xticks=range(3), xticklabels=order, ylabel=r'Relative PSD bin sum (Hz$^{-1}$)')
    ax.tick_params(axis='x', rotation=20)
    save(fig, panel)
    plt.show()
