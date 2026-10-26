from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parents[1]/'data'/'screening'
COLORS = {'opChRmine': '#cf3b41', 'opChroME2s': '#e8a044',
          'CsChrimson': '#477fba', 'opKCR1-C29D': '#477fba', 'control': '#333333'}
OPSINS = ['opChRmine', 'opChroME2s', 'CsChrimson', 'control']
POWERS = {22: 200, 45: 300, 77: 500}
POWER_CODES = {10: 150, **POWERS}
INHIBITION_WINDOW = slice(20, 23)


def configure():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'legend.frameon': False})


def load(neuron, side, opsin, power_mw):
    with np.load(DATA/f'{neuron}_{side}_{opsin}_{POWER_CODES[power_mw]}.npz', allow_pickle=False) as f:
        values = {key: f[key] for key in f.files}
    reducer = np.nanmean if neuron == 'PPL101' else np.mean
    rows = []
    for i in range(len(values['fly_ids'])):
        trials = values['trials_dff'][values['trial_fly_index'] == i]
        rows.append(reducer(trials - np.nanmean(trials[:, :10]), axis=0))
    values['fly_mean_dff'] = np.asarray(rows)
    return values


def plot_mean(ax, rows, color, label=None, time=None):
    rows = np.asarray(rows)
    if time is None:
        time = (np.arange(rows.shape[-1])-19)/12
    mean = np.nanmean(rows, axis=0)
    sem = stats.sem(rows, axis=0, nan_policy='omit')
    ax.plot(time, mean, color=color, label=label)
    ax.fill_between(time, mean-sem, mean+sem, color=color, alpha=.2, linewidth=0)
    ax.axvspan(0, 1/12, color='#d17697', alpha=.25, linewidth=0)
    ax.set(xlabel='Time from stimulation (s)', ylabel=r'$\Delta F/F_0$')


def fig1_h():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
    for ax, power in zip(axes, POWERS):
        for opsin in OPSINS:
            data = load('MBON11', 'ipsi', opsin, power)
            plot_mean(ax, data['fly_mean_dff'], COLORS[opsin], f'{opsin} (n={len(data["fly_ids"])})')
        ax.set_title(f'Fig. 1H: {power} mW')
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    return fig


def metric_table():
    rows = []
    for neuron, side, opsins, powers in [('MBON11', 'ipsi', OPSINS, list(POWERS)),
                                      ('MBON11', 'contra', OPSINS, [22]),
                                      ('PPL101', 'ipsi', ['opKCR1-C29D','control'], [10,22,45])]:
        for opsin in opsins:
            for power in powers:
                data = load(neuron, side, opsin, power)
                for fly, trace in zip(data['fly_ids'], data['fly_mean_dff']):
                    if neuron == 'MBON11' and side == 'ipsi':
                        metrics = {'peak_frames20_39': np.max(trace[20:40]), 'legacy_plot_frame20':trace[20]}
                    else:
                        metrics = {'trough_frames20_22': np.min(trace[INHIBITION_WINDOW])}
                    for metric, value in metrics.items():
                        rows.append(dict(neuron=neuron,side=side,opsin=opsin,power_mw=power,
                                         fly_id=fly,metric=metric,value=value))
    return pd.DataFrame(rows)


def fig1_i(metric='peak_frames20_39'):
    table = metric_table().query('neuron == "MBON11" and side == "ipsi" and metric == @metric')
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    for opsin in OPSINS:
        group = table[table.opsin == opsin].groupby('power_mw').value
        ax.errorbar(group.mean().index, group.mean(), group.sem(), color=COLORS[opsin], marker='o', label=opsin)
    ax.set(xlabel='Power (mW)', ylabel=r'Peak $\Delta F/F_0$', title=f'Fig. 1I: {metric}')
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def fig1_k():
    fig, ax = plt.subplots(figsize=(4.5, 3.3))
    for opsin in OPSINS:
        data = load('MBON11','contra',opsin,22)
        plot_mean(ax,data['fly_mean_dff'],COLORS[opsin],f'{opsin} (n={len(data["fly_ids"])})')
    ax.legend(fontsize=7)
    ax.set_title('Fig. 1K: contralateral activation, 22 mW')
    fig.tight_layout()
    return fig


def fig1_l():
    values = metric_table().query('neuron == "MBON11" and side == "contra" and metric == "trough_frames20_22"')
    fig, ax = plt.subplots(figsize=(4.5, 3.3))
    groups=[]
    for i,opsin in enumerate(['control','CsChrimson','opChroME2s','opChRmine']):
        v=values.loc[values.opsin==opsin,'value'].to_numpy()
        groups.append(v)
        ax.bar(i,np.mean(v),color=COLORS[opsin],alpha=.5)
        ax.errorbar(i,np.mean(v),stats.sem(v),color='black',capsize=3)
        ax.scatter(i+np.linspace(-.12,.12,len(v)),v,s=14,color=COLORS[opsin],edgecolors='white',linewidths=.3)
    ax.set_xticks(range(4),['control','CsChrimson','opChroME2s','opChRmine'],rotation=25)
    ax.set(ylabel=r'$\Delta F/F_0$',title='Fig. 1L')
    fig.tight_layout()
    return fig, stats.f_oneway(*groups)


def fig1_o_p():
    fig, axes = plt.subplots(1,2,figsize=(7.5,3),sharey=True)
    for ax,power in zip(axes,[22,45]):
        for opsin in ['opKCR1-C29D','control']:
            plot_mean(ax,load('PPL101','ipsi',opsin,power)['fly_mean_dff'],COLORS[opsin],opsin)
        ax.set_title(f'Fig. 1O: {power} mW')
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig_p,ax=plt.subplots(figsize=(4,3))
    table=metric_table().query('neuron == "PPL101" and metric == "trough_frames20_22"')
    for opsin in ['opKCR1-C29D','control']:
        v=table[table.opsin==opsin].groupby('power_mw').value
        ax.errorbar(v.mean().index,v.mean(),v.sem(),marker='o',color=COLORS[opsin],label=opsin)
    ax.set(title='Fig. 1P',xlabel='Power (mW)',ylabel=r'$\Delta F/F_0$')
    ax.legend(fontsize=7)
    fig_p.tight_layout()
    return fig,fig_p


def fig1_p_statistics():
    table = metric_table().query('neuron == "PPL101" and metric == "trough_frames20_22"')
    wide = table.pivot(index=['opsin', 'fly_id'], columns='power_mw', values='value')
    y = wide.to_numpy()
    if list(wide.columns) != [10, 22, 45] or not np.isfinite(y).all():
        raise ValueError('Each fly must have a finite response at all three powers.')
    group_curves = wide.groupby(level='opsin').transform('mean').to_numpy()
    n, k = y.shape
    g = wide.index.get_level_values('opsin').nunique()
    grand = y.mean()
    group_means = group_curves.mean(axis=1, keepdims=True)
    power_effect = y.mean(axis=0, keepdims=True) - grand
    residual = y - group_curves
    within_residual = residual - residual.mean(axis=1, keepdims=True)

    ss = np.array([
        k * np.sum((group_means - grand) ** 2),
        n * np.sum(power_effect ** 2),
        np.sum((group_curves - group_means - power_effect) ** 2),
    ])
    error_between = k * np.sum((y.mean(axis=1, keepdims=True) - group_means) ** 2)
    error_within = np.sum(within_residual ** 2)
    df1 = np.array([g - 1, k - 1, (g - 1) * (k - 1)])
    df2 = np.array([n - g, (n - g) * (k - 1), (n - g) * (k - 1)])
    f = (ss / df1) / (np.array([error_between, error_within, error_within]) / df2)

    covariance = within_residual.T @ within_residual / (n - g)
    epsilon = np.clip(np.trace(covariance) ** 2 / ((k - 1) * np.sum(covariance ** 2)),
                      1 / (k - 1), 1)
    correction = np.array([1, epsilon, epsilon])
    return pd.DataFrame({
        'effect': ['opsin', 'power', 'opsin:power'],
        'F': f, 'df1': df1, 'df2': df2, 'p': stats.f.sf(f, df1, df2),
        'epsilon': correction,
        'p_GG': stats.f.sf(f, df1 * correction, df2 * correction),
    })


def plot_examples(kind='image', powers=(22,45,77)):
    fig, axes = plt.subplots(3,len(powers),figsize=(3*len(powers),6),squeeze=False)
    manifest=json.loads((DATA/'source_manifest.json').read_text())
    for row,opsin in enumerate(OPSINS[:3]):
        for col,power in enumerate(powers):
            spec=next(s for s in manifest['examples'] if s['opsin']==opsin and s['power_mw']==power)
            with np.load(DATA/f'example_{opsin}_{power}mW.npz',allow_pickle=False) as a:
                ax=axes[row,col]
                if kind=='image':
                    im=ax.imshow(np.concatenate([a['pre_dff'],a['stim_dff']],axis=1),cmap='jet',vmin=0,vmax=1.4)
                    ax.set_xticks([125,375],['pre','stim']);ax.set_yticks([])
                else:
                    width=spec['trace_smoothing_frames']; y=np.convolve(a['trace_dff'],np.ones(width)/width,mode='same')
                    ax.plot(np.arange(len(y))/12,y,color=COLORS[opsin])
                    for start in [19,79,139]:ax.axvspan(start/12,(start+1)/12,color='#d17697',alpha=.3)
                    ax.set(xlabel='Time (s)',ylabel=r'$\Delta F/F_0$')
                ax.set_title(f'{opsin}, {power} mW',fontsize=8)
    if kind == 'image':
        fig.tight_layout(rect=(0,0,.92,1))
        cax = fig.add_axes([.94,.23,.016,.52])
        fig.colorbar(im,cax=cax,label=r'$\Delta F/F_0$')
    else:
        fig.tight_layout()
    return fig


def plot_single_animals(powers=(22,45,77)):
    fig, axes=plt.subplots(3,len(powers),figsize=(3*len(powers),6),squeeze=False)
    for row,(opsin,fly_index) in enumerate([('opChRmine',9),('opChroME2s',3),('CsChrimson',3)]):
        for col,power in enumerate(powers):
            data=load('MBON11','ipsi',opsin,power)
            trace=data['continuous_mean_dff'][fly_index]
            y=np.convolve(trace,np.ones(3)/3,mode='same')
            ax=axes[row,col]
            ax.plot(np.arange(len(y))/12,y,color=COLORS[opsin])
            for start in [79,139,199]:ax.axvspan(start/12,(start+1)/12,color='#d17697',alpha=.3)
            ax.set(title=f'{opsin}, {power} mW',xlabel='Time (s)',ylabel=r'$\Delta F/F_0$')
    fig.tight_layout()
    return fig


def fig1_n():
    data=load('PPL101','ipsi','opKCR1-C29D',45)
    y=np.convolve(data['continuous_mean_dff'][4],np.ones(3)/3,mode='same')
    fig,ax=plt.subplots(figsize=(5,2.6))
    ax.plot(np.arange(len(y))/12,y,color=COLORS['opKCR1-C29D'])
    for start in [79,139,199]:ax.axvspan(start/12,(start+1)/12,color='#d17697',alpha=.3)
    ax.set(title='Fig. 1N: PPL101 example fly',xlabel='Time (s)',ylabel=r'$\Delta F/F_0$')
    fig.tight_layout()
    return fig


def save(fig, output, name):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fig.savefig(output/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(output/f'{name}.svg',bbox_inches='tight')
    return fig
