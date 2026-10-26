import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from epg_analysis import DATA, EXCITATION, FS, POWER_MW
from epg_panels import configure_font, save

COLORS = dict(zip(EXCITATION, ('#208FC5', '#DE861D', '#C82342')))
MARKERS = {4: 'o', 10: 's', 22: '^', 45: 'D'}


def circular_fwhm(profile):
    """Circular FWHM in radians, using the global peak and spatial minimum."""
    values = np.asarray(profile, dtype=float)
    if values.shape != (16,) or not np.isfinite(values).all():
        raise ValueError('Expected a finite spatial profile with 16 ROIs.')
    peak = int(values.argmax())
    if values[peak] <= values.min():
        return np.nan
    threshold = (values[peak] + values.min()) / 2
    crossings = []
    for direction in (-1, 1):
        for step in range(1, 17):
            inside = values[(peak + direction * (step - 1)) % 16]
            outside = values[(peak + direction * step) % 16]
            if outside <= threshold:
                crossings.append(step - 1 + (inside - threshold) / (inside - outside))
                break
    return sum(crossings) * (2 * np.pi / 16) if len(crossings) == 2 else np.nan


def measurements():
    rows = []
    for group in EXCITATION:
        for code, power in POWER_MW.items():
            with np.load(DATA / 'excitation' / f'{group}_{code}.npz') as data:
                profiles = data['each_fly_16roi'][:, :, 20:40].sum(axis=-1) / FS
                ids = data['fly_ids'].astype(str)
            for fly, profile in zip(ids, profiles):
                peak = int(profile.argmax())
                rows.append(dict(
                    group=group, fly_id=fly, power_mw=power,
                    response_auc_dff_seconds=float(profile[[3, 4]].mean()),
                    width_fwhm_rad=circular_fwhm(profile),
                    peak_roi_1based=peak + 1,
                    target_associated=peak in (2, 3, 4, 5),
                ))
    return pd.DataFrame(rows)


def summary(observations=None):
    observations = measurements() if observations is None else observations
    selected = observations.loc[observations.target_associated]
    return selected.groupby(['group', 'power_mw'], sort=False).agg(
        n_flies=('fly_id', 'size'),
        response_mean=('response_auc_dff_seconds', 'mean'),
        response_sem=('response_auc_dff_seconds', 'sem'),
        width_mean=('width_fwhm_rad', 'mean'),
        width_sem=('width_fwhm_rad', 'sem'),
    ).reset_index()


def plot():
    values = summary()
    style = {'font.family': configure_font(), 'font.size': 9,
             'axes.spines.top': False, 'axes.spines.right': False,
             'axes.linewidth': .8, 'legend.frameon': False,
             'pdf.fonttype': 42, 'svg.fonttype': 'none'}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(7.2, 3.3))
        fig.subplots_adjust(left=.12, right=.69, bottom=.22, top=.95)
        for group in EXCITATION:
            points = values.loc[values.group.eq(group)].sort_values('power_mw')
            ax.plot(points.response_mean, points.width_mean, color=COLORS[group], lw=1.1)
            for row in points.itertuples(index=False):
                ax.errorbar(row.response_mean, row.width_mean,
                            xerr=row.response_sem, yerr=row.width_sem,
                            marker=MARKERS[row.power_mw], color=COLORS[group],
                            markeredgecolor='white', markeredgewidth=.45,
                            markersize=5.5, elinewidth=.75, capsize=2, linestyle='none')
        ax.set(xlim=(0, 2.4), ylim=(np.pi / 6, 1.65), xticks=[0, 1, 2],
               yticks=[np.pi / 6, np.pi / 3, np.pi / 2],
               yticklabels=['π/6', 'π/3', 'π/2'],
               xlabel=r'Response magnitude (AUC, $\Delta F/F_0$ · s)',
               ylabel='Bump width (FWHM, rad)')
        group_handles = [Line2D([], [], color=COLORS[group], lw=1.3, label=group)
                         for group in EXCITATION]
        fig.legend(handles=group_handles, loc='upper left',
                   bbox_to_anchor=(.72, .94), borderaxespad=0)
        power_handles = [Line2D([], [], color='.4', marker=marker, linestyle='none',
                                markersize=5, label=f'{power} mW')
                         for power, marker in MARKERS.items()]
        fig.legend(handles=power_handles, loc='lower left',
                   bbox_to_anchor=(.72, .24), borderaxespad=0)
        save(fig, 'Fig5F_response_width')
        plt.show()
        plt.close(fig)
