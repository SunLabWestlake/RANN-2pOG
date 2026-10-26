from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'epg_images'
OUT = ROOT / 'outputs' / 'epg_images'
QUALITY = {}


def show(panel):
    manifest = pd.read_csv(DATA / 'manifest.csv')
    order = {'CsChrimson': 0, 'opChroME2s': 1, 'opChRmine': 2, 'opKCR1-C29D': 3}
    selected = manifest[manifest.panel.eq(panel)].sort_values('group', key=lambda s: s.map(order))
    if selected.empty:
        raise ValueError(f'No representative fluorescence images for {panel}')
    OUT.mkdir(parents=True, exist_ok=True)
    quality = []
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42}):
        for row in selected.to_dict('records'):
            path = DATA / row['cache_file']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
            with np.load(path, allow_pickle=False) as data:
                baseline = gaussian_filter(data['baseline_mean'], sigma=row['gaussian_sigma_px'])
                valid = baseline != 0
                pre = np.divide(data['pre_mean'] - baseline, baseline,
                                out=np.full_like(baseline, np.nan), where=valid)
                response = np.divide(data['stim_mean'] - baseline, baseline,
                                     out=np.full_like(baseline, np.nan), where=valid)
            name = f"{row['panel']}_{row['group']}"
            quality.append(dict(panel=row['panel'], group=row['group'],
                                zero_baseline_pixels=int((~valid).sum()),
                                total_pixels=baseline.size))
            fig, ax = plt.subplots(figsize=(3.2, 3.0))
            im = ax.imshow(response, vmin=row['display_vmin'],
                           vmax=row['display_vmax'], cmap='jet', interpolation='nearest')
            ax.set_title(f"{row['panel']}: {row['group']}")
            ax.axis('off')
            fig.colorbar(im, ax=ax, shrink=.75, label='ΔF/F₀')
            fig.tight_layout()
            for extension in ('svg', 'pdf', 'png'):
                fig.savefig(OUT / f'{name}.{extension}', dpi=200, bbox_inches='tight')
            plt.show()
    QUALITY[panel] = pd.DataFrame(quality)
