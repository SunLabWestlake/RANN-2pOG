from pathlib import Path
from functools import lru_cache
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'analysis'
PAIRS = {
    "0": (3, 4), "pi_8": (2, 5), "pi_4": (1, 6),
    "3pi_8": (0, 7), "pi_2": (15, 8), "5pi_8": (14, 9),
    "3pi_4": (13, 10), "7pi_8": (12, 11),
}
POWER_MW = {100: 4, 150: 10, 200: 22, 300: 45}
FS = 12.0
EXCITATION = ("CsChrimson", "opChroME2s", "opChRmine")
COLORS = {"CsChrimson": "#00a0e9", "opChroME2s": "#f39800",
          "opChRmine": "#c30d23", "opKCR1-C29D": "#22ac38"}
COMPETITION_KINETIC_ROIS = {'a': 4, 'b': 12}
COMPETITION_KINETIC_PANELS = {'a': 'J', 'b': 'I'}


def merge_distance_by_fly(each_fly, fly_ids):
    """(fly, 16 ROI, time) -> (8 distance bins, fly, time)."""
    x = np.asarray(each_fly, dtype=float)
    ids = [str(i) for i in fly_ids]
    if x.ndim != 3 or x.shape[1] != 16:
        raise ValueError(f"Expected (fly,16,time), got {x.shape}")
    if len(ids) != x.shape[0] or len(set(ids)) != len(ids):
        raise ValueError("Exactly one unique fly_id is required per input row.")
    return np.stack([np.mean(x[:, pair, :], axis=1)
                     for pair in PAIRS.values()], axis=0)


def normalize_mean_auc(auc):
    """Group-mean endpoint scaling; apply the same offset and range to each fly."""
    x = np.asarray(auc, dtype=float)
    if x.ndim != 2 or x.shape[0] != len(PAIRS) or x.shape[1] < 2:
        raise ValueError('Expected AUC shaped (8 distance bins, at least 2 flies).')
    if not np.isfinite(x).all():
        raise ValueError('Mean-profile normalization requires finite fly AUC values.')
    mean = x.mean(axis=1)
    offset = mean[-1]
    span = mean[0] - offset
    if np.isclose(span, 0):
        raise ValueError('Group-mean endpoint normalization has a zero range.')
    return dict(mean=(mean - offset) / span,
                sem=stats.sem(x, axis=1, ddof=1) / abs(span),
                fly_values=(x - offset) / span,
                n_flies=x.shape[1])


def normalize_auc(auc):
    x = np.asarray(auc, dtype=float)
    low, high = x.min(axis=0, keepdims=True), x.max(axis=0, keepdims=True)
    def divide(numerator, denominator):
        return np.divide(numerator, denominator,
            out=np.full_like(numerator, np.nan), where=~np.isclose(denominator, 0))
    return {"minmax": divide(x - low, high - low),
            "first_last": normalize_mean_auc(x)['fly_values'],
            "by_stimulated": divide(x, x[:1])}


def metrics_from_merged(merged, group):
    """Antipodal minimum in frames 20:30; latency is relative to frame 19."""
    stim, anti = merged[0], merged[-1]
    anti_frame = 20 + anti[:, 20:30].argmin(axis=1)
    out = {
        "stim_peak_dff_full60": stim.max(axis=1),
        "stim_peak_time_s_frame19": (stim.argmax(axis=1) - 19) / FS,
        "antipodal_trough_dff_frames20_29": np.take_along_axis(
            anti, anti_frame[:, None], axis=1)[:, 0],
        "antipodal_trough_time_s_frame19": (anti_frame - 19) / FS,
        "stim_auc_sum_frames20_39": stim[:, 20:40].sum(axis=1),
    }
    if group == "opKCR1-C29D":
        out.update(stim_trough_dff_full60=stim.min(axis=1),
            antipodal_peak_dff_full60=anti.max(axis=1))
    return out


def rm_anova_gg(values):
    """Repeated-measures ANOVA with GG correction; rows are flies, columns are powers."""
    all_values = np.asarray(values, dtype=float)
    y = all_values[np.isfinite(all_values).all(axis=1)]
    n, k = y.shape
    result = dict(n_flies=n, n_excluded=all_values.shape[0] - n,
                  F=np.nan, df1=k - 1, df2=(n - 1) * (k - 1),
                  epsilon_gg=np.nan, p_uncorrected=np.nan, p_gg=np.nan,
                  status="not_testable")
    if n < 3 or k < 2:
        return result
    centered = y - y.mean(axis=1, keepdims=True)
    if np.max(np.abs(centered)) < 1e-12:
        result["status"] = "no_within_fly_variation"
        return result
    residual = centered - centered.mean(axis=0, keepdims=True)
    ss_effect = n * np.square(centered.mean(axis=0)).sum()
    ss_error = np.square(residual).sum()
    if ss_error <= np.finfo(float).eps * max(np.square(centered).sum(), 1e-30):
        result["status"] = "zero_residual_variance"
        return result
    f = (ss_effect / (k - 1)) / (ss_error / ((n - 1) * (k - 1)))
    covariance = np.cov(centered, rowvar=False, ddof=1)
    epsilon = np.trace(covariance) ** 2 / ((k - 1) * np.trace(covariance @ covariance))
    epsilon = float(np.clip(epsilon, 1 / (k - 1), 1))
    result.update(F=float(f), epsilon_gg=epsilon,
        p_uncorrected=float(stats.f.sf(f, k - 1, (n - 1) * (k - 1))),
        p_gg=float(stats.f.sf(f, epsilon * (k - 1), epsilon * (n - 1) * (k - 1))),
        status="ok")
    return result


def add_holm(frame, p_col, group_cols, output_col, planned_family_size=None):
    result = frame.copy()
    result[output_col] = np.nan
    for _, group in result.groupby(group_cols, sort=False, dropna=False):
        valid = group[p_col].notna()
        p = group[p_col].fillna(1).to_numpy()
        if planned_family_size is not None:
            p = np.r_[p, np.ones(max(0, planned_family_size - len(p)))]
        adjusted = multipletests(p, method="holm")[1][:len(group)]
        result.loc[group.index[valid], output_col] = adjusted[valid.to_numpy()]
    return result


def power_comparisons(auc_table):
    rows = []
    for (group, normalization, bin_name), frame in auc_table.groupby(
            ["group", "normalization", "distance_bin"], sort=False):
        pivot = frame.pivot(index="fly_id", columns="power_code", values="value")
        pivot = pivot.reindex(columns=list(POWER_MW))
        rows.append(dict(group=group, normalization=normalization,
            distance_bin=bin_name, **rm_anova_gg(pivot.to_numpy())))
    return add_holm(pd.DataFrame(rows), "p_gg", ["group", "normalization"],
                    "p_gg_holm_8_bins", planned_family_size=8)


def mixed_anova(values):
    """Mixed ANOVA: opsin between flies, time within fly; GG correction."""
    arrays = [np.asarray(a, float) for a in values]
    assert all(a.ndim == 2 and np.isfinite(a).all() for a in arrays)
    y = np.vstack(arrays)
    n, k = y.shape
    g = len(arrays)
    grand = y.mean()
    time_mean = y.mean(axis=0)
    ss_group = k*sum(len(a)*(a.mean()-grand)**2 for a in arrays)
    ss_subject = k*sum(np.square(a.mean(axis=1)-a.mean()).sum() for a in arrays)
    ss_time = n*np.square(time_mean-grand).sum()
    ss_interaction = sum(len(a)*np.square(a.mean(axis=0)-a.mean()-time_mean+grand).sum() for a in arrays)
    residuals = [a-a.mean(axis=1, keepdims=True)-a.mean(axis=0, keepdims=True)+a.mean() for a in arrays]
    ss_error = sum(np.square(e).sum() for e in residuals)
    covariance = sum(e.T@e for e in residuals)/(n-g)
    eps = np.clip(np.trace(covariance)**2/((k-1)*np.trace(covariance@covariance)), 1/(k-1), 1)
    rows = []
    for label, ss, dfn, den, dfd, correction in [
        ('opsin', ss_group, g-1, ss_subject, n-g, 1.),
        ('time', ss_time, k-1, ss_error, (n-g)*(k-1), eps),
        ('opsin × time', ss_interaction, (g-1)*(k-1), ss_error, (n-g)*(k-1), eps)]:
        f = (ss/dfn)/(den/dfd)
        rows.append(dict(effect=label, n_flies=n, F=f, df1=dfn, df2=dfd,
                         epsilon_GG=correction, p_uncorrected=stats.f.sf(f, dfn, dfd),
                         p_GG=stats.f.sf(f, dfn*correction, dfd*correction)))
    return pd.DataFrame(rows)


@lru_cache(None)
def single(group, power=150):
    study = 'inhibition' if group == 'opKCR1-C29D' else 'excitation'
    with np.load(DATA / study / f'{group}_{power}.npz', allow_pickle=False) as data:
        raw, ids = data['each_fly_16roi'].copy(), data['fly_ids'].copy()
    if raw.shape != (len(ids), 16, 60) or not np.isfinite(raw).all():
        raise ValueError(f'Invalid fly-level input for {group}, {power}')
    return merge_distance_by_fly(raw, ids), ids


@lru_cache(None)
def competition(group, normalization='per_frame_minmax'):
    with np.load(DATA / 'competing' / f'{group}.npz', allow_pickle=False) as data:
        raw, ids = data['raw_16roi_fly_time'].copy(), data['fly_ids'].copy()
    if raw.shape != (len(ids), 16, 90) or not np.isfinite(raw).all():
        raise ValueError(f'Invalid competition input for {group}')
    if len(set(ids)) != len(ids):
        raise ValueError('Fly IDs must be unique within a group')
    if normalization == 'per_frame_minmax':
        lower, upper = raw.min(axis=1, keepdims=True), raw.max(axis=1, keepdims=True)
        raw = (raw - lower) / (upper - lower + 1e-8)
    elif normalization != 'raw_dff':
        raise ValueError('normalization must be per_frame_minmax or raw_dff')
    a = np.stack([raw[:, :, start:start+5] for start in range(20,70,10)], axis=1).mean(axis=1)
    b = np.stack([raw[:, :, start:start+5] for start in range(25,75,10)], axis=1).mean(axis=1)
    return dict(a=a, b=b, ids=ids, full=raw,
        region_a_contrast=a[:, [3,4], :].mean(axis=1)-b[:, [3,4], :].mean(axis=1),
        region_b_contrast=b[:, [11,12], :].mean(axis=1)-a[:, [11,12], :].mean(axis=1))


def auc_profiles():
    rows = []
    for group in EXCITATION:
        reference_ids = single(group, 100)[1]
        for power in POWER_MW:
            x, ids = single(group, power)
            np.testing.assert_array_equal(ids, reference_ids)
            auc = x[:, :, 20:40].sum(axis=2)
            for scale, values in {'raw_sum_frames':auc, 'raw_integral_seconds':auc/FS,
                                  **normalize_auc(auc)}.items():
                for bi, bin_name in enumerate(PAIRS):
                    rows.extend(dict(group=group, fly_id=fly, power_code=power,
                        power_mw=POWER_MW[power], distance_bin=bin_name,
                        normalization=scale, value=float(value)) for fly, value in zip(ids,values[bi]))
    return pd.DataFrame(rows)


def competition_kinetic_values(group, region='a', normalization='raw_dff',
                               curve_normalization=None):
    """Own-minus-other bouts per fly; optional scaling uses the group-mean baseline and peak."""
    if region not in COMPETITION_KINETIC_ROIS:
        raise ValueError('region must be a or b')
    data = competition(group, normalization)
    roi = COMPETITION_KINETIC_ROIS[region]
    other = 'b' if region == 'a' else 'a'
    values = data[region][:, roi, :] - data[other][:, roi, :]
    if curve_normalization is None:
        return values
    if curve_normalization != 'baseline_peak':
        raise ValueError('curve_normalization must be None or baseline_peak')
    mean = values.mean(axis=0)
    baseline = mean[0]
    span = mean.max() - baseline
    if not np.isfinite(values).all() or np.isclose(span, 0):
        raise ValueError('Mean-curve normalization requires a finite, nonzero peak range.')
    return (values - baseline) / span


def competition_pointwise_tests(curve_normalization=None, normalization='raw_dff'):
    rows = []
    for region in COMPETITION_KINETIC_ROIS:
        for frame in range(5):
            for ga,gb in combinations(EXCITATION,2):
                a,b=[competition_kinetic_values(g, region, normalization, curve_normalization)[:,frame]
                     for g in [ga,gb]]
                test=stats.mannwhitneyu(a,b,alternative='two-sided',method='auto')
                rows.append(dict(normalization=normalization,curve_normalization=curve_normalization,
                    panel=COMPETITION_KINETIC_PANELS[region],region=region,
                    roi_1based=COMPETITION_KINETIC_ROIS[region]+1,frame=frame,time_s=frame/FS,
                    group_a=ga,group_b=gb,n_flies_a=len(a),n_flies_b=len(b),
                    U=float(test.statistic),p_raw=float(test.pvalue)))
    return add_holm(pd.DataFrame(rows),'p_raw',['normalization','region'],'p_holm_15')


def competition_spatial_tests():
    rows=[]
    for scale in ['raw_dff','per_frame_minmax']:
        for group in EXCITATION:
            d=competition(group,scale)
            differences=d['a'].mean(axis=2)-d['b'].mean(axis=2)
            for roi in range(16):
                values=differences[:,roi]
                if np.allclose(values,0):
                    w,p=0.,1.
                else:
                    test=stats.wilcoxon(values,alternative='two-sided',method='auto')
                    w,p=float(test.statistic),float(test.pvalue)
                rows.append(dict(normalization=scale,group=group,roi_1based=roi+1,
                    n_flies=len(values),W=w,p_raw=p))
    return add_holm(pd.DataFrame(rows),'p_raw',['normalization','group'],'p_holm_16')
