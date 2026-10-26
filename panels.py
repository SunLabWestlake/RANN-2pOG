"""Figure panels for the EPG population models."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm, Normalize
from . import workflow as wf

DISTANCE_COLORS = plt.get_cmap('RdBu')(np.linspace(0, 1, 8))
SHADES = {'CsChrimson': ['#bdd7e7', '#6baed6', '#00aeef'],
          'ChroME2s': ['#fdd0a2', '#fdae6b', '#ff8c00'],
          'ChRmine': ['#f2b8b0', '#e67f73', '#c8172b']}
DATASET_LABELS = {'uniform': 'Uniform inhibition', 'hemibrain': 'Hemibrain',
                  'banc': 'BANC', 'malecns': 'MaleCNS'}


class ModelPanels:
    def __init__(self, recompute=False):
        self.recompute = recompute
        self.results = {}
        plt.rcParams.update({'font.size': 8, 'font.family': 'sans-serif',
                             'font.sans-serif': ['Arial', 'DejaVu Sans'],
                             'pdf.fonttype': 42, 'svg.fonttype': 'none',
                             'axes.spines.top': False, 'axes.spines.right': False})

    def result(self, dataset, opsin, condition='mid'):
        key = (dataset, opsin, condition)
        if key not in self.results:
            loader = wf.simulate if self.recompute else wf.load_result
            self.results[key] = loader(opsin, condition, dataset=dataset)
        return self.results[key]

    @staticmethod
    def _display(fig):
        plt.show()
        plt.close(fig)

    @staticmethod
    def _stimulation(ax, condition):
        times = np.arange(0, 50, 5) / 12 if condition == 'repetitive' else [0]
        for time in times:
            ax.axvspan(time, time + .05, color='#f0a1ac', alpha=.6, lw=0)

    def heatmaps(self, dataset):
        fig, axes = plt.subplots(1, 4, figsize=(10, 2.2), layout='constrained')
        for ax, opsin in zip(axes, wf.OPSINS):
            data = self.result(dataset, opsin)
            mean, _ = wf.summary(data['activity_16'])
            time = data['times_s'] - 1.6
            norm = Normalize(-.11, .05) if opsin == 'KCR' else AsinhNorm(.1, vmin=-.16, vmax=2.4)
            im = ax.imshow(mean[:, 10:], cmap='jet', norm=norm, aspect='auto',
                           interpolation='nearest', extent=(time[10] - 1 / 24, time[-1] + 1 / 24, 7.5, -.5))
            ax.set(title=wf.LABELS[opsin], xlabel='Time (s)', yticks=[0, 4, 7],
                   yticklabels=['0', 'π/2', '7π/8'])
            ticks = [-.11, 0, .05] if opsin == 'KCR' else [-.16, 0, .4, 2.4]
            bar = fig.colorbar(im, ax=ax, orientation='horizontal', label='Δactivity (a.u.)',
                               shrink=.85, ticks=ticks)
            bar.ax.set_xticklabels([f'{tick:g}' for tick in ticks])
            bar.minorticks_off()
        axes[0].set_ylabel('Distance to stimulated ROI')
        self._display(fig)

    def distance_traces(self, dataset, condition='mid', opsins=None):
        opsins = wf.OPSINS if opsins is None else tuple(opsins)
        fig, axes = plt.subplots(1, len(opsins), figsize=(2.6 * len(opsins), 2.5),
                                 squeeze=False, layout='constrained')
        for ax, opsin in zip(axes[0], opsins):
            data = self.result(dataset, opsin, condition)
            mean, _ = wf.summary(data['activity_16'])
            time = data['times_s'] - wf.specification(opsin, condition)['onset_s']
            for values, color in zip(mean, DISTANCE_COLORS):
                ax.plot(time, values, color=color, lw=1.2)
            self._stimulation(ax, condition)
            ax.set(title=f'{DATASET_LABELS[dataset]} · {wf.LABELS[opsin]}', xlabel='Time (s)')
            if condition == 'repetitive':
                ax.set_ylim(-.9, 7.2)
            else:
                ax.set_ylim((-.112, .056) if opsin == 'KCR' else (-.37, 2.63))
        axes[0, 0].set_ylabel('Δactivity (a.u.)')
        self._display(fig)

    def kernels(self):
        kernels = wf.load_kernels()
        fig, ax = plt.subplots(figsize=(4.2, 2.6), layout='constrained')
        colors = dict(ChRmine='#c8172b', ChroME2s='#ff8c00', CsChrimson='#00aeef', KCR='#149448')
        for opsin in wf.OPSINS:
            values = kernels[opsin]
            ax.plot(np.arange(len(values)) * .01, values, color=colors[opsin],
                    label=wf.LABELS[opsin], lw=1.3)
        ax.set(xlabel='Time after stimulation (s)', ylabel='Normalized input kernel')
        ax.legend(frameon=False)
        self._display(fig)

    def spatial_profiles(self, dataset):
        fig, axes = plt.subplots(1, 3, figsize=(8, 2.8), sharey=True, layout='constrained')
        for ax, opsin in zip(axes, ('CsChrimson', 'ChroME2s', 'ChRmine')):
            for kind, label, color in zip(('low', 'mid', 'high'), ('low', 'middle', 'high'), SHADES[opsin]):
                data = self.result(dataset, opsin, kind)
                _, normalized = wf.normalized_spatial_auc(data['activity_16'])
                ax.plot(wf.DISTANCE_PI, normalized, '-o', ms=3, lw=1.2, color=color, label=label)
            ax.set(title=wf.LABELS[opsin], xlabel='Distance to stimulated ROI',
                   xticks=[0, .25, .5, .75], xticklabels=['0', 'π/4', 'π/2', '3π/4'],
                   ylim=(-.05, 1.05), yticks=[0, .5, 1])
            ax.legend(frameon=False)
        axes[0].set_ylabel('Normalized response magnitude (AUC)')
        self._display(fig)
