from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from epg_analysis import (POWER_MW, EXCITATION, COLORS, FS,
    COMPETITION_KINETIC_ROIS, COMPETITION_KINETIC_PANELS,
    ROOT, normalize_mean_auc, metrics_from_merged, mixed_anova,
    single, competition, competition_kinetic_values, auc_profiles, power_comparisons,
    competition_pointwise_tests, competition_spatial_tests)

OUT = ROOT / 'outputs' / 'analysis'
FIGURES = OUT / 'figures'
RESULTS = {}
GROUPS = list(EXCITATION)
SHORT = dict(zip(GROUPS, ['CsChrimson', 'opChroME2s', 'opChRmine']))
SHORT['opKCR1-C29D'] = 'opKCR1-C29D'
ANGLE_LABELS = ['0', 'π/8', 'π/4', '3π/8', 'π/2', '5π/8', '3π/4', '7π/8']
TIME = (np.arange(60) - 19) / FS
METRIC_LABELS = {
    'stim_peak_dff_full60': 'Stimulated peak ΔF/F₀',
    'stim_peak_time_s_frame19': 'Time to peak (s)',
    'antipodal_trough_dff_frames20_29': 'Antipodal trough ΔF/F₀',
    'antipodal_trough_time_s_frame19': 'Time to trough (s)',
    'stim_auc_sum_frames20_39': 'Stimulated AUC (ΔF/F₀ × frames)',
}
MAIN_METRIC_PANELS = {
    'stim_peak_dff_full60': 'Fig. 2G',
    'stim_peak_time_s_frame19': 'Fig. 2I',
    'antipodal_trough_dff_frames20_29': 'Fig. 4B',
    'antipodal_trough_time_s_frame19': 'Fig. 4D',
}


def configure_font():
    try:
        font_manager.findfont('Arial', fallback_to_default=False)
        return 'Arial'
    except ValueError:
        return 'DejaVu Sans'


def table(frame, name):
    RESULTS[name] = frame.copy()
    return frame


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for label in fig.findobj(match=Text):
        if '₀' in label.get_text():
            label.set_text(label.get_text().replace('₀', '$_0$'))
    fly_points = [(ax, points) for ax in fig.axes for points in ax.collections
                  if points.get_gid() == 'fly-observations']
    if fly_points:
        fig.canvas.draw()
        for ax, points in fly_points:
            left, right = ax.transData.transform([(0, 0), (.55, 0)])[:, 0]
            bar_width_pt = abs(right-left) * 72 / fig.dpi
            axes_height_pt = ax.get_window_extent().height * 72 / fig.dpi
            outer_diameter_pt = min(.19 * bar_width_pt, .05 * axes_height_pt)
            outline_pt = .10 * outer_diameter_pt
            points.set_sizes([(outer_diameter_pt-outline_pt)**2])
            points.set_linewidths([outline_pt])
    for suffix in ['svg', 'pdf', 'png']:
        fig.savefig(FIGURES / f'{name}.{suffix}', dpi=160, bbox_inches='tight')


def line(ax, x, color, label, t=TIME, *, show_sem=True):
    avg = np.mean(x, axis=0)
    se = stats.sem(x, axis=0, ddof=1)
    ax.plot(t, avg, color=color, label=label, lw=1.5)
    if show_sem:
        ax.fill_between(t, avg-se, avg+se, color=color, alpha=.18, lw=0)


def trace_axes(ax, ylabel='ΔF/F₀'):
    ax.axhline(0, color='.7', lw=.5)
    ax.axvspan(0, .05, ymin=0, ymax=.045, color='#d978ad', lw=0)
    ax.set(xlabel='Time (s)', ylabel=ylabel)
    ax.legend(fontsize=7)


def distance(group, power=150, view='trace', output_name=None):
    x, ids = single(group, power)
    fig, ax = plt.subplots(figsize=(5.0, 3.0), layout='constrained')
    if view == 'heatmap':
        limits = (-.2, .2) if group == 'opKCR1-C29D' else (-.1, .7)
        im = ax.imshow(x.mean(axis=1), aspect='auto', cmap='jet',
                       extent=[TIME[0], TIME[-1], 7.5, -.5], vmin=limits[0], vmax=limits[1])
        ax.set(yticks=range(8), yticklabels=ANGLE_LABELS, xlabel='Time (s)',
               ylabel='Angular distance')
        fig.colorbar(im, ax=ax, label='ΔF/F₀')
    else:
        from spatial_panels import DISTANCE_COLORS
        for b, label in enumerate(ANGLE_LABELS):
            line(ax, x[b], DISTANCE_COLORS[b], label)
        trace_axes(ax)
    ax.set_title(f'{group} · {POWER_MW[power]} mW')
    save(fig, output_name or f'{group}_{power}_{view}')


def metric_values(metric, group, power=150):
    x, _ = single(group, power)
    return metrics_from_merged(x, group)[metric]


def metric_summary(metric, group=None):
    rows = []
    for g in ([group] if group else GROUPS):
        v = metric_values(metric, g)
        rows.append(dict(group=g, n_flies=len(v), mean=v.mean(), sem=stats.sem(v)))
    return pd.DataFrame(rows)


def significance_label(p):
    if not np.isfinite(p):
        return 'NA'
    for threshold, label in [(1e-4, '****'), (1e-3, '***'), (.01, '**'), (.05, '*')]:
        if p < threshold:
            return label
    return 'ns'


def metric_test(metric, groups=None):
    rows = []
    for a, b in combinations(GROUPS, 2):
        va, vb = metric_values(metric, a), metric_values(metric, b)
        test = stats.mannwhitneyu(va, vb, alternative='two-sided', method='auto')
        rows.append(dict(group_a=a, group_b=b, n_flies_a=len(va), n_flies_b=len(vb),
                         U=test.statistic, p_raw=test.pvalue))
    df = pd.DataFrame(rows)
    df['p_Holm_3_pairs'] = multipletests(df.p_raw, method='holm')[1]
    if metric in MAIN_METRIC_PANELS:
        df['significance_raw'] = df.p_raw.map(significance_label)
        df['significance_Holm'] = df.p_Holm_3_pairs.map(significance_label)
        df = df[['group_a', 'group_b', 'n_flies_a', 'n_flies_b', 'U',
                 'p_raw', 'significance_raw', 'p_Holm_3_pairs', 'significance_Holm']]
    if groups:
        df = df[df.apply(lambda row: {row.group_a, row.group_b} == set(groups), axis=1)]
    return table(df, f'{metric}_tests'+('_'+'_'.join(groups) if groups else ''))


def mean_bar_with_fly_points(ax, position, values, color, offsets):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    mean = finite.mean()
    sem = stats.sem(finite, ddof=1)
    ax.bar(position, mean, width=.55, color=color, alpha=.23,
           edgecolor='none', zorder=1)
    points = ax.scatter(position+.5*np.asarray(offsets), values, s=72, facecolors='black',
                        edgecolors='white', linewidths=.9, alpha=1, zorder=3, clip_on=False)
    points.set_gid('fly-observations')
    ax.errorbar(position, mean, yerr=sem, fmt='none', capsize=4,
                color='black', elinewidth=1.1, capthick=1.1, zorder=4)


def metric(metric, group=None, mode='plot', groups=None):
    if mode == 'test':
        metric_test(metric, groups)
        return
    table(metric_summary(metric, group), f'{metric}_summary'+('_'+group if group else ''))
    if mode == 'summary':
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.4), layout='constrained')
    rng = np.random.default_rng(20261012)
    for j, g in enumerate(GROUPS):
        values = metric_values(metric, g)
        mean_bar_with_fly_points(ax, j, values, COLORS[g], rng.uniform(-.12, .12, len(values)))
    ax.axhline(0, color='.65', lw=.6, zorder=0)
    ax.set(xticks=range(3), xticklabels=[f'{SHORT[g]}\nn={len(single(g)[1])}' for g in GROUPS],
           ylabel=METRIC_LABELS[metric], title='10 mW · one point per fly · mean ± SEM')
    save(fig, metric)
    metric_test(metric)


def overlay(region=0, normalized=False):
    fig, ax = plt.subplots(figsize=(5.4, 3.2), layout='constrained')
    for g in GROUPS:
        x, ids = single(g)
        v = x[region].copy()
        if normalized:
            mean = v.mean(axis=0)
            scale = mean.max() if region == 0 else abs(mean.min())
            if np.isclose(scale, 0):
                raise ValueError('Zero group-mean normalization denominator')
            v /= scale
        line(ax, v, COLORS[g], g, show_sem=not normalized)
    trace_axes(ax, 'Normalized ΔF/F₀' if normalized else 'ΔF/F₀')
    save(fig, f'overlay_region{region}_normalized{normalized}')


def power_profiles(group=None, view='spatial'):
    gs = [group] if group else GROUPS
    if view == 'spatial':
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), layout='constrained')
        for ax, (power, mw) in zip(axes, POWER_MW.items()):
            for g in gs:
                x, _ = single(g, power)
                auc = x[:, :, 20:40].sum(axis=2)
                ax.errorbar(range(8), auc.mean(axis=1), yerr=stats.sem(auc, axis=1),
                            color=COLORS[g], label=g, capsize=2, marker='o', ms=2.5)
            ax.set(xticks=range(8), xticklabels=ANGLE_LABELS, xlabel='Angular distance',
                   ylabel='AUC (ΔF/F₀ × frames)', title=f'{mw} mW')
            ax.legend(fontsize=7)
    else:
        fig, axes = plt.subplots(1, len(gs), figsize=(4.4*len(gs), 3.2),
                                 squeeze=False, layout='constrained')
        for ax, g in zip(axes[0], gs):
            rgb = np.array(plt.matplotlib.colors.to_rgb(COLORS[g]))
            for j, power in enumerate(POWER_MW):
                x, _ = single(g, power)
                color = tuple(1-(1-rgb)*(.35+.21*j))
                if view == 'traces':
                    line(ax, x[0], color, f'{POWER_MW[power]} mW')
                elif view == 'normalized':
                    profile = normalize_mean_auc(x[:, :, 20:40].sum(axis=2))
                    ax.errorbar(range(8), profile['mean'], yerr=profile['sem'],
                                color=color, label=f'{POWER_MW[power]} mW',
                                capsize=2, marker='o', ms=2.5)
                else:
                    raise ValueError(view)
            if view == 'traces':
                trace_axes(ax)
            else:
                ax.set(xticks=range(8), xticklabels=ANGLE_LABELS, xlabel='Angular distance',
                       ylabel='Normalized AUC')
                ax.legend(fontsize=7)
            ax.set_title(g)
    save(fig, f'Fig5_power_{view}_'+(group or 'all'))
    if view == 'normalized':
        df = power_comparisons(auc_profiles())
        table(df[df.normalization.eq('first_last') & df.group.isin(gs)], 'Fig5E_RM_ANOVA')


def normalized_spatial():
    fig, ax = plt.subplots(figsize=(5.5, 3.1), layout='constrained')
    for g in GROUPS:
        x, ids = single(g, 200)
        profile = normalize_mean_auc(x[:, :, 20:40].sum(axis=2))
        ax.errorbar(range(8), profile['mean'], yerr=profile['sem'],
                    label=f'{g} (n={len(ids)})', color=COLORS[g], marker='o', ms=3, capsize=2)
    ax.set(xticks=range(8), xticklabels=ANGLE_LABELS, xlabel='Angular distance',
           ylabel='Endpoint-normalized AUC', title='Fig. 5D · 22 mW · fly mean ± SEM')
    ax.legend(fontsize=7)
    save(fig, 'Fig5D_endpoint_normalized')


def power_auc():
    fig, ax = plt.subplots(figsize=(5.3, 3.2), layout='constrained')
    for g in GROUPS:
        v = np.column_stack([metric_values('stim_auc_sum_frames20_39', g, p) for p in POWER_MW])
        ax.errorbar(range(4), v.mean(axis=0), yerr=stats.sem(v, axis=0),
                    marker='o', ms=3, color=COLORS[g], label=f'{g} (n={len(v)})', capsize=2)
    ax.set(xticks=range(4), xticklabels=list(POWER_MW.values()),
           xlabel='Power (mW)', ylabel='AUC (ΔF/F₀ × frames)')
    ax.legend(fontsize=7)
    save(fig, 'Fig5C_stimulated_auc')




def competition_kinetics(region='a', normalization='raw_dff', group=None,
                         mode='plot', curve_normalization='baseline_peak', show_sem=False):
    gs = [group] if group else GROUPS
    arrays = [competition_kinetic_values(g, region, normalization, curve_normalization) for g in gs]
    panel = COMPETITION_KINETIC_PANELS[region]
    other = 'b' if region == 'a' else 'a'
    stem = f'Fig6{panel}_ROI{region}_kinetics_'+(group or 'all')
    if mode == 'values':
        rows = [dict(panel=panel, region=region, roi_1based=COMPETITION_KINETIC_ROIS[region]+1,
                     group=g, fly_id=fly, **{f'frame_{j}': row[j] for j in range(5)})
                for g, x in zip(gs, arrays) for fly, row in zip(competition(g, normalization)['ids'], x)]
        table(pd.DataFrame(rows), stem+'_fly_values')
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.3), layout='constrained')
    for g, x in zip(gs, arrays):
        line(ax, x, COLORS[g], f'{g} (n={len(x)})', t=np.arange(5)/FS, show_sem=show_sem)
    ax.axhline(0, color='.7', lw=.5)
    ax.axvspan(0, .05, ymin=0, ymax=.045, color='#d978ad', lw=0)
    normalized = curve_normalization is not None or normalization != 'raw_dff'
    ax.set(xlabel='Time from bout onset (s)', ylabel=('Normalized' if normalized else 'ΔF/F₀')+' contrast',
           title=f'Fig. 6{panel} · ROI {region} · stimulation {region} − {other}')
    ax.legend(fontsize=7)
    save(fig, stem)
    if group is None:
        table(mixed_anova(arrays), stem+'_mixed_ANOVA')
        df = competition_pointwise_tests(curve_normalization=curve_normalization,
                                         normalization=normalization)
        table(df[df.region.eq(region) & df.normalization.eq(normalization)],
              stem+'_pointwise_MWU')


def competition_spatial(group=None, mode='profiles'):
    gs = [group] if group else GROUPS
    if mode in ['profiles', 'contrast']:
        fig, ax = plt.subplots(figsize=(6.2, 3.1), layout='constrained')
        for g in gs:
            data = competition(g)
            a, b = data['a'].mean(axis=2), data['b'].mean(axis=2)
            if mode == 'profiles':
                for values, color, label in [(a, '#9b59b6', 'Region a'), (b, '#377eb8', 'Region b')]:
                    mean, sem = values.mean(axis=0), stats.sem(values, axis=0)
                    roi = np.arange(1, 17)
                    ax.plot(mean, roi, color=color, label=label, lw=1.3)
                    ax.fill_betweenx(roi, mean-sem, mean+sem, color=color, alpha=.18, lw=0)
            else:
                line(ax, a-b, COLORS[g], f'{g} (n={len(a)})', t=np.arange(1, 17))
        if mode == 'profiles':
            fig.set_size_inches(3.4, 4.5)
            ax.set(xlabel='Normalized response', ylabel='ROI', ylim=(16.5, .5),
                   yticks=[1, 4, 5, 8, 12, 13, 16], title=group)
        else:
            ax.set(xlabel='ROI', ylabel='Normalized contrast', xticks=[1, 4, 5, 8, 12, 13, 16])
        ax.legend(fontsize=7)
        save(fig, f'Fig6_spatial_{mode}_'+(group or 'all'))
    elif mode == 'wilcoxon':
        df = competition_spatial_tests()
        table(df[df.group.isin(gs) & df.normalization.eq('per_frame_minmax')], f'Fig6_spatial_wilcoxon_'+(group or 'all'))
    elif mode == 'anova':
        rows = []
        arrays = [competition(g)['a'].mean(axis=2)-competition(g)['b'].mean(axis=2) for g in GROUPS]
        for name, rois in [(f'ROI_{j+1}', [j]) for j in range(16)]+[('region_a_pair', [3, 4]), ('region_b_pair', [11, 12])]:
            vals = [a[:, rois].mean(axis=1) for a in arrays]
            test = stats.f_oneway(*vals)
            rows.append(dict(position=name, n_CsChrimson=len(vals[0]), n_opChroME2s=len(vals[1]),
                             n_opChRmine=len(vals[2]), F=test.statistic, p_raw=test.pvalue))
        df = pd.DataFrame(rows)
        df['p_Holm_within_family'] = np.r_[multipletests(df.p_raw[:16], method='holm')[1],
                                                          multipletests(df.p_raw[16:], method='holm')[1]]
        df['family'] = ['16 positions']*16+['2 averaged regions']*2
        table(df, 'Fig6H_fly_ANOVA_positions_and_paired_regions')


def competition_heatmap(group):
    data = competition(group, 'raw_dff')
    mean = data['full'].mean(axis=0)
    fig, ax = plt.subplots(figsize=(6.4, 3.0), layout='constrained')
    im = ax.imshow(mean, aspect='auto', origin='upper', cmap='jet',
        extent=[0, 90/FS, 16.5, .5])
    for start in range(20,70,10):
        ax.axvspan(start/FS,(start+5)/FS,ymin=0,ymax=.035,color='#9b59b6',lw=0)
        ax.axvspan((start+5)/FS,(start+10)/FS,ymin=0,ymax=.035,color='#377eb8',lw=0)
    ax.set(xlabel='Time from original frame 60 (s)', ylabel='Original file ROI (1-based)',
        title=group, yticks=[1,4,5,8,12,13,16])
    fig.colorbar(im, ax=ax, label='Mean ΔF/F₀')
    save(fig, f'Fig6_heatmap_{group}')


def show(panel, **kwargs):
    family = configure_font()
    plt.rcParams.update({'font.family': family, 'font.size': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    functions = {'competition_heatmap': competition_heatmap, 'distance': distance,
                 'metric': metric, 'overlay': overlay, 'power_profiles': power_profiles,
                 'normalized_spatial': normalized_spatial, 'power_auc': power_auc,
                 'competition_kinetics': competition_kinetics, 'competition_spatial': competition_spatial}
    functions[panel](**kwargs)
    plt.show()


