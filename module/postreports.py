"""
    Consolidate outputs from the selection and counterfactuals stages into the
    additional tables and figures needed for the manuscript Discussion section.

    Neither stage's own report script produces these directly: selreport.py writes
    one summary table per metric, not the single multi-metric table the manuscript
    uses; cfreports.py writes one report per patient instance, not the cross-patient
    aggregates. This script reads their already-generated per-run artifacts (CSVs,
    joblib benchmarking stats) and combines them.

    Adapted from the ad hoc aggregation notebook `legacy/202608 notebooks/cfreports.ipynb`,
    updated for the per-patient file/folder naming (patient_code-based, e.g.
    `catboost_split0_patient029_local_cf_distances.csv` under `split0/nofiltering/029/`)
    that the current cfreports.py writes -- the notebook's version predates patient_code
    tracking and located instances by raw dataframe index instead.

    Usage: python postreports.py <config file>
    e.g.:  python postreports.py bin_postreport_final_202608.yml
"""
import sys
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

import ymlconfig


# Same red/blue pair the per-patient counterfactual heatmaps use, so the two
# figure families stay visually consistent.
CF_DECREASE = '#B31B2E'
CF_INCREASE = '#2367AC'

SELECTION_METRIC_COLUMNS = {
    'auprc': 'AUPRC',
    'roc-auc': 'ROC-AUC',
    'sensitivity': 'Sensitivity',
    'specificity': 'Specificity',
    'accuracy': 'Accuracy',
    'precision': 'Precision',
    'youden': 'Youden Index',
    'f1': 'F1 Score',
}


def consolidate_selection(config, config_path, outputdir):
    """Algorithm x metric summary table for the selected feature set (matches the
    manuscript's model-selection table), sorted by AUPRC descending. Each cell is
    formatted as 'mean +/- std' across the repeated k-fold runs."""
    tag = config.selection.tag
    feature_set = config.selection.feature_set
    stats_path = (config_path / 'binary' / 'selection' / tag / 'benchmarking'
                  / f'{feature_set}_benchmarking_stats.joblib')
    stats = joblib.load(stats_path)

    metrics = list(config.selection.metrics)
    mean_table = stats['mean'][metrics].rename(columns=SELECTION_METRIC_COLUMNS)
    std_table = stats['std'][metrics].rename(columns=SELECTION_METRIC_COLUMNS)

    order = mean_table['AUPRC'].sort_values(ascending=False).index
    mean_table = mean_table.loc[order]
    std_table = std_table.loc[order]

    table = mean_table.astype(object)
    for col in mean_table.columns:
        table[col] = [f'{m:.3f} ± {s:.3f}' for m, s in zip(mean_table[col], std_table[col])]
    table.index.name = 'Algorithm'
    table = table.reset_index()

    sel_dir = outputdir / 'selection'
    sel_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(sel_dir / 'selection_metrics_summary.csv', index=False)

    # At 3-decimal mean+/-std over 8 metrics, the table is wider than main.tex's
    # textwidth (372pt) at any reasonable portrait font/spacing -- verified by
    # test-compiling against manuscript/sn-jnl.cls (overflowed by ~188pt at
    # \footnotesize). A sidewaystable (main.tex already imports the `rotating`
    # package) is the only configuration confirmed to fit, with room to spare.
    body = table.to_latex(index=False)
    latex = (
        '\\begin{sidewaystable}\n'
        '\\centering\n'
        '\\footnotesize\n'
        '\\setlength{\\tabcolsep}{4pt}\n'
        + body +
        '\\caption{Aggregate performance metrics (mean $\\pm$ std) for all classification '
        'algorithms evaluated on the full feature set under repeated stratified k-fold '
        'cross-validation.}\n'
        '\\label{tab:selection_metrics_meanstd}\n'
        '\\end{sidewaystable}\n'
    # '±' -> '$\pm$': to_latex would otherwise emit the raw UTF-8 character,
    # which pdflatex's default font encoding cannot typeset.
    ).replace('±', '$\\pm$')
    with open(sel_dir / 'selection_metrics_summary.latex', 'w') as f:
        f.write(latex)
    print(f'Wrote {sel_dir / "selection_metrics_summary.csv"} ({feature_set} feature set, tag={tag})')
    return table


def _outcome(actual, pred):
    if actual and pred:
        return 'True Positive'
    if not actual and pred:
        return 'False Positive'
    if not actual and not pred:
        return 'True Negative'
    return 'False Negative'


def _plot_global_cf_heatmap(df, path, title, vmin, vmax, fmt=".2f"):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df, annot=True, fmt=fmt, cmap="viridis", linewidths=0.5,
                vmin=vmin, vmax=vmax, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def _read_cf_distances(dist_path):
    """One patient's counterfactual distances, without the query instance.

    generate_diverse_cfs pools its counterfactuals into a frame whose first row is the
    query instance itself ("include instance values in the report"), and that row
    survives into the distances file as cf_row 0 with sparsity/L1/L2 all zero. It is
    not a counterfactual: counting it inflates every CF Count by one, and averaging it
    in pulls every sparsity/L1/L2 mean toward zero. Drop it here rather than at the
    source, so the per-patient artifacts keep showing the baseline they are read
    against.
    """
    dist_df = pd.read_csv(dist_path).set_index('cf_row')
    query = dist_df.loc[0]
    if query.sparsity != 0:
        raise ValueError(f'{dist_path}: cf_row 0 was expected to be the query instance '
                         f'(sparsity 0) but changes {int(query.sparsity)} features')
    return dist_df.drop(index=0)


def consolidate_counterfactuals(config, config_path, outputdir):
    """Cross-model instance-level and aggregate counterfactual tables/figures:
    - cf_fulltable: one row per patient instance that produced counterfactuals
    - ioi_summary_per_model: candidate/misclassified/borderline counts per model,
      with mean sparsity/L1/L2 over instances that produced counterfactuals
    - cf_changed_features: how often each actionable feature was changed, per instance
    - global_cf_percent.png / global_cf_counts.png: feature x model change heatmaps
    """
    model_code = config.counterfactuals.model.code
    tag = config.counterfactuals.tag
    nsplits = config.counterfactuals.nsplits
    actionable_features = config.counterfactuals.actionable_features.split(',')

    cf_dir = config_path / 'binary' / 'counterfactuals' / model_code / tag

    instance_rows = []
    changed_rows = []
    model_summary_rows = []

    for midx in range(nsplits):
        split_dir = cf_dir / f'split{midx}'
        ioi_path = split_dir / f'{model_code}_split{midx}_instances_of_interest.csv'
        ioi_df = pd.read_csv(ioi_path)

        n_with_cf = 0
        sparsity_means, l1_means, l2_means = [], [], []

        for _, instance in ioi_df.iterrows():
            code = int(instance.patient_code)
            patient_dir = split_dir / 'nofiltering' / f'{code:03d}'
            prefix = f'{model_code}_split{midx}_patient{code:03d}'
            dist_path = patient_dir / f'{prefix}_local_cf_distances.csv'
            changed_path = patient_dir / f'{prefix}_local_cf_most_changed.csv'
            if not dist_path.is_file():
                # candidate instance never produced usable counterfactuals
                continue

            dist_df = _read_cf_distances(dist_path)
            cf_count = len(dist_df)
            sparsity_mean = dist_df['sparsity'].mean()
            l1_mean = dist_df['L1_dist'].mean()
            l2_mean = dist_df['L2_dist'].mean()

            n_with_cf += 1
            sparsity_means.append(sparsity_mean)
            l1_means.append(l1_mean)
            l2_means.append(l2_mean)

            instance_rows.append({
                'Model': midx,
                'Patient Code': code,
                'Probability': instance.pred_proba,
                'Margin': instance.margin,
                'Outcome': _outcome(instance.actual, instance.pred),
                'CF Count': cf_count,
                'Sparsity Mean': sparsity_mean,
                'L1 Mean': l1_mean,
                'L2 Mean': l2_mean,
            })

            if changed_path.is_file():
                counts = pd.read_csv(changed_path).set_index('feature')['change count']
                row = {'Model': midx, 'Patient Code': code, 'CF Count': cf_count}
                row.update({f: int(counts.get(f, 0)) for f in actionable_features})
                changed_rows.append(row)

        model_summary_rows.append({
            'Model': midx,
            'Candidates': len(ioi_df),
            'Misclassified': int(ioi_df.misclassified.sum()),
            # Candidates that were not misclassified. Every one of these is within
            # threshold_delta of its fold's threshold (a candidate that is not
            # misclassified can only have qualified via the borderline rule), so this
            # and 'Misclassified' partition 'Candidates'. It is NOT the count of all
            # low-confidence candidates: misclassified ones can be borderline too.
            'Borderline (correct)': int((~ioi_df.misclassified).sum()),
            'Instances with CF': n_with_cf,
            'Sparsity Mean': np.mean(sparsity_means) if sparsity_means else np.nan,
            'L1 Mean': np.mean(l1_means) if l1_means else np.nan,
            'L2 Mean': np.mean(l2_means) if l2_means else np.nan,
        })

    cf_out_dir = outputdir / 'counterfactuals'
    cf_out_dir.mkdir(parents=True, exist_ok=True)

    # Instance-level full table
    instance_table = pd.DataFrame(instance_rows)
    instance_table.to_csv(cf_out_dir / 'cf_fulltable.csv', index=False)
    with open(cf_out_dir / 'cf_fulltable.latex', 'w') as f:
        f.write(instance_table.to_latex(index=False, float_format="{:.2f}".format))

    # Per-model summary table
    model_summary = pd.DataFrame(model_summary_rows)
    model_summary.to_csv(cf_out_dir / 'ioi_summary_per_model.csv', index=False)
    with open(cf_out_dir / 'ioi_summary_per_model.latex', 'w') as f:
        f.write(model_summary.to_latex(index=False, float_format="{:.2f}".format))

    # Feature-change counts per instance, with a Total row
    changed_table = pd.DataFrame(changed_rows)
    total_row = {'Model': '', 'Patient Code': 'Total'}
    total_row.update(changed_table[['CF Count'] + actionable_features].sum().to_dict())
    changed_table = pd.concat([changed_table, pd.DataFrame([total_row])], ignore_index=True)
    changed_table.to_csv(cf_out_dir / 'cf_changed_features.csv', index=False)
    with open(cf_out_dir / 'cf_changed_features.latex', 'w') as f:
        f.write(changed_table.to_latex(index=False, float_format="{:.0f}".format))

    # Global feature-change heatmaps: fraction (and raw count) of each model's
    # counterfactuals that changed a given actionable feature
    changed_only = pd.DataFrame(changed_rows)
    percent_by_model, counts_by_model = {}, {}
    for midx, group in changed_only.groupby('Model'):
        total_cfs = group['CF Count'].sum()
        counts_by_model[midx] = group[actionable_features].sum()
        percent_by_model[midx] = (
            group[actionable_features].sum() / total_cfs if total_cfs
            else group[actionable_features].sum().astype(float)
        )

    percent_df = pd.DataFrame(percent_by_model)
    percent_df = percent_df.loc[percent_df.mean(axis=1).sort_values(ascending=False).index]
    _plot_global_cf_heatmap(
        percent_df, cf_out_dir / 'global_cf_percent.png',
        title='Counterfactuals - Global Feature Importance (fraction of model CFs)',
        vmin=0, vmax=1)

    counts_df = pd.DataFrame(counts_by_model).astype(int)
    counts_df = counts_df.loc[counts_df.mean(axis=1).sort_values(ascending=False).index]
    _plot_global_cf_heatmap(
        counts_df, cf_out_dir / 'global_cf_counts.png',
        title='Counterfactuals - Global Feature Importance (counts)',
        vmin=counts_df.values.min(), vmax=counts_df.values.max(), fmt='d')

    print(f'Wrote {cf_out_dir} (model={model_code}, tag={tag}, {len(instance_table)} instances with CFs)')
    return instance_table, model_summary, changed_table


def _find_instance(cf_dir, model_code, nsplits, patient_code):
    """Locate which fold's held-out test set a patient belongs to.

    Returns (split_index, instance_row). Patients are not distributed by any rule the
    caller can compute -- the fold assignment comes from the stratified k-fold split --
    so the only reliable way to find one is to scan each split's candidate list.
    """
    for midx in range(nsplits):
        ioi_path = (cf_dir / f'split{midx}'
                    / f'{model_code}_split{midx}_instances_of_interest.csv')
        ioi_df = pd.read_csv(ioi_path)
        match = ioi_df[ioi_df.patient_code == patient_code]
        if len(match):
            return midx, match.iloc[0]
    raise ValueError(f'patient {patient_code} is not a counterfactual candidate in '
                     f'any of the {nsplits} splits under {cf_dir}')


def _load_cf_stage_config(cf_dir):
    """Load the counterfactual stage's own config from the copy it left in its outputs.

    Every stage copies its config next to the artifacts it produced, so the settings
    that generated those artifacts are recoverable here. Reading them back is preferable
    to restating them in this stage's config: a restated value is only correct until
    someone re-runs the counterfactual stage with a different one, and nothing would
    flag the divergence.
    """
    configs = sorted(cf_dir.glob('*.yml'))
    if len(configs) != 1:
        raise ValueError(f'expected exactly one copied stage config in {cf_dir}, '
                         f'found {len(configs)}: {[c.name for c in configs]}')
    return ymlconfig.dict_to_namespace(ymlconfig.load_config(configs[0]))


def _load_fold_thresholds(config_path, cf_config, model_code):
    """Per-fold decision thresholds as the optimization stage published them.

    Anchored on the trained-models path the counterfactual stage was pointed at, so
    these are the thresholds belonging to the very models that produced the
    counterfactuals being plotted, not merely those of a same-named run.
    """
    opt_dir = Path(cf_config.optimization.first_repeat_trained_models_filename).parent
    metrics = pd.read_csv(
        config_path / opt_dir / f'{model_code}_first_repeat_optimization_metrics.csv')
    return metrics.set_index('split')['threshold']


def _load_case_changes(cf_dir, model_code, midx, patient_code, actionable_features):
    """Read one patient's counterfactuals and return (baseline, change matrix).

    Row 0 of `<prefix>_local_cf.csv` is the query instance itself and the remainder are
    its counterfactuals, so the changes are rows 1..n minus row 0, restricted to the
    actionable features (the file carries every column, but only these six can vary).

    Rows are ordered by how many features they change, then by the size of the HbA1c
    change, which groups the single-feature counterfactuals at the top and makes the
    HbA1c panel read monotonically instead of in DiCE's arbitrary generation order.
    """
    prefix = f'{model_code}_split{midx}_patient{patient_code:03d}'
    cf_path = (cf_dir / f'split{midx}' / 'nofiltering' / f'{patient_code:03d}'
               / f'{prefix}_local_cf.csv')
    df = pd.read_csv(cf_path)
    baseline = df.iloc[0][actionable_features].astype(float)
    changes = df.iloc[1:][actionable_features].astype(float).values - baseline.values
    hba1c_col = actionable_features.index('HBA1C')
    order = np.lexsort((changes[:, hba1c_col], (changes != 0).sum(axis=1)))
    return baseline, changes[order]


def _cf_panel_figure(config, config_path, outputdir, case_codes, filename,
                     row_height=0.055, panel_pad=0.78, hspace=0.55):
    """One combined figure holding a panel per patient in `case_codes`.

    Replaces the per-patient full-page reports that `cfreports.py`'s
    `plot_local_cf_heatmap2` writes (those stay, as the per-patient diagnostic for all
    successful instances); this is the manuscript figure, and it differs in three ways
    that matter for reading it:

    - Several patients share one figure instead of one full page each.
    - HbA1c gets a real quantitative axis rather than a colour, and the axis is shared
      across the panels, so HbA1c changes are comparable *between* patients. The
      per-patient heatmap could not support that comparison: it gave each patient its
      own colour scale.
    - Binary features get direction glyphs rather than the same colour ramp as HbA1c.
      A comorbidity flip and a change in HbA1c percentage points are different units,
      and the shared diverging colormap made a 1 -> 0 insulin flip look like a larger
      effect than a -0.23 point HbA1c change.

    The patient's own profile is deliberately not drawn on the figure: the manuscript
    states it in each case's prose, and the per-fold threshold and margin are in the
    instance-level table, so repeating them here only cost space.
    """
    model_code = config.counterfactuals.model.code
    tag = config.counterfactuals.tag
    nsplits = config.counterfactuals.nsplits
    actionable_features = config.counterfactuals.actionable_features.split(',')

    binary_features = [f for f in actionable_features if f != 'HBA1C']
    hba1c_col = actionable_features.index('HBA1C')
    cf_dir = config_path / 'binary' / 'counterfactuals' / model_code / tag

    # The rule that made a candidate 'borderline' belongs to the counterfactual stage,
    # so its own config -- not a copy of the number here -- decides which panels are
    # labeled Borderline and which Confident.
    cf_config = _load_cf_stage_config(cf_dir)
    borderline_delta = cf_config.dice.threshold_delta
    fold_thresholds = _load_fold_thresholds(config_path, cf_config, model_code)

    cases = []
    for code in case_codes:
        midx, instance = _find_instance(cf_dir, model_code, nsplits, code)
        baseline, changes = _load_case_changes(
            cf_dir, model_code, midx, code, actionable_features)
        # The threshold is not stored alongside the instance, but margin is defined as
        # the absolute distance from it, so it is recoverable from the side the
        # prediction fell on. That inversion is exact by construction (both `pred` and
        # `margin` come from the same threshold in get_instances_of_interest), but it
        # is an assumption reaching across two stages' file formats, so check it
        # against the published value rather than letting a future divergence pass as
        # a plausible-looking number on a figure.
        threshold = (instance.pred_proba - instance.margin if instance.pred
                     else instance.pred_proba + instance.margin)
        published = fold_thresholds[midx]
        if not np.isclose(threshold, published, atol=1e-6):
            raise ValueError(
                f'patient {code}: threshold recovered from split {midx} candidate list '
                f'({threshold:.6f}) disagrees with the optimization stage threshold '
                f'({published:.6f}); the two stages have diverged')
        confidence = 'Borderline' if instance.margin <= borderline_delta else 'Confident'
        outcome = f'{confidence} {_outcome(instance.actual, instance.pred).lower()}'
        cases.append(dict(code=code, midx=midx, baseline=baseline, changes=changes,
                          outcome=outcome, pred_proba=instance.pred_proba,
                          threshold=threshold))

    # One shared HbA1c axis across panels, so the panels can be compared to each other.
    deltas = np.concatenate([c['changes'][:, hba1c_col] for c in cases])
    lo, hi = min(deltas.min(), 0), max(deltas.max(), 0)
    pad = 0.12 * (hi - lo)

    # Panels are proportional to their counterfactual counts, but the figure height is
    # per-row plus a fixed allowance per panel (title, x-axis labels) and one for the
    # legend -- scaling the whole figure by the row count instead would leave a short
    # panel's labels overlapping and a tall one's rows absurdly far apart.
    heights = [len(c['changes']) for c in cases]
    fig_height = row_height * sum(heights) + panel_pad * len(cases) + 0.45
    fig = plt.figure(figsize=(6.5, fig_height))
    gs = fig.add_gridspec(len(cases), 2, height_ratios=heights,
                          width_ratios=[1.15, 1], hspace=hspace, wspace=0.14)

    for i, case in enumerate(cases):
        changes = case['changes']
        n = len(changes)
        rows = np.arange(n)

        # --- HbA1c: signed magnitude on a shared axis
        ax_hba1c = fig.add_subplot(gs[i, 0])
        delta = changes[:, hba1c_col]
        colors = [CF_DECREASE if v < 0 else CF_INCREASE for v in delta]
        ax_hba1c.hlines(rows, 0, delta, color=colors, linewidth=1.5)
        ax_hba1c.scatter(delta, rows, s=8, color=colors, zorder=3)
        ax_hba1c.axvline(0, color='#666666', linewidth=0.8)
        ax_hba1c.set_xlim(lo - pad, hi + pad)
        ax_hba1c.set_ylim(n - 0.5, -0.5)
        ax_hba1c.set_yticks([])
        ax_hba1c.tick_params(axis='x', labelsize=6)
        ax_hba1c.set_ylabel(f'{n} CFs', fontsize=6.5)
        ax_hba1c.set_xlabel(
            f"HbA1c change from {case['baseline']['HBA1C']:.4g}% (percentage points)",
            fontsize=6.5, labelpad=1)
        for side in ('top', 'right'):
            ax_hba1c.spines[side].set_visible(False)

        # --- Binary features: direction of the flip, one column per feature
        ax_flips = fig.add_subplot(gs[i, 1])
        for j, feature in enumerate(binary_features):
            values = changes[:, actionable_features.index(feature)]
            introduced, resolved = values > 0, values < 0
            ax_flips.scatter(np.full(introduced.sum(), j), rows[introduced],
                             marker='^', s=22, color=CF_INCREASE)
            ax_flips.scatter(np.full(resolved.sum(), j), rows[resolved],
                             marker='v', s=22, color=CF_DECREASE)
        ax_flips.set_xlim(-0.6, len(binary_features) - 0.4)
        ax_flips.set_ylim(n - 0.5, -0.5)
        ax_flips.set_xticks(range(len(binary_features)))
        ax_flips.set_xticklabels(
            [f"{f}\n({case['baseline'][f]:.0f})" for f in binary_features], fontsize=6)
        ax_flips.set_yticks([])
        ax_flips.grid(axis='x', color='#E8E8E8', linewidth=0.6)
        ax_flips.set_axisbelow(True)
        for side in ('top', 'right', 'left'):
            ax_flips.spines[side].set_visible(False)

        ax_hba1c.text(
            0, 1.10,
            f"({chr(ord('a') + i)}) Patient {case['code']:03d} · Model {case['midx']} "
            f"· {case['outcome']} · p = {case['pred_proba']:.3f}, "
            f"threshold {case['threshold']:.3f}",
            transform=ax_hba1c.transAxes, fontsize=8, va='bottom')

    fig.legend(handles=[
        Line2D([], [], marker='^', linestyle='', color=CF_INCREASE,
               label='0 $\\to$ 1  introduced / started'),
        Line2D([], [], marker='v', linestyle='', color=CF_DECREASE,
               label='1 $\\to$ 0  resolved / stopped')],
        loc='lower center', ncol=2, fontsize=7, frameon=False,
        bbox_to_anchor=(0.5, -0.035))

    cf_out_dir = outputdir / 'counterfactuals'
    cf_out_dir.mkdir(parents=True, exist_ok=True)
    path = cf_out_dir / filename
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {path} (patients {", ".join(str(c) for c in case_codes)})')
    return path


def _case_study_codes(config):
    return [int(c) for c in str(config.counterfactuals.case_studies).split(',')]


def plot_case_study_counterfactuals(config, config_path, outputdir):
    """The manuscript figure: the patients the main text discusses individually."""
    return _cf_panel_figure(config, config_path, outputdir, _case_study_codes(config),
                            'case_study_counterfactuals.png')


def plot_remaining_counterfactuals(config, config_path, outputdir, instance_table):
    """The same figure for the successful instances the main text does not discuss.

    The appendix companion to `plot_case_study_counterfactuals`: every instance that
    produced counterfactuals but was not selected as a case study, so between the two
    figures the manuscript shows all of them. The membership is derived from the
    instance table rather than listed in the config, so selecting a different case
    study moves that patient between the two figures instead of dropping it from both
    or showing it twice.

    Panels here share their own HbA1c axis, not the case-study figure's: the two
    figures are read separately, and a scale stretched to cover both would flatten the
    differences within each.
    """
    case_codes = _case_study_codes(config)
    remaining = [int(c) for c in instance_table['Patient Code'] if c not in case_codes]
    # Five panels rather than three, and the manuscript places this one on a page of
    # its own, so the rows are drawn tighter to keep the figure plus its caption within
    # a single text height. `hspace` is raised to compensate: it is a fraction of the
    # mean panel height, so shorter panels would otherwise shrink the gap between them
    # below what a panel's x-axis label and the next panel's title need, while the
    # space those two need does not shrink with the panels.
    return _cf_panel_figure(config, config_path, outputdir, remaining,
                            'remaining_counterfactuals.png',
                            row_height=0.048, hspace=0.75)


def main():
    if len(sys.argv) < 2:
        print("Usage: python postreports.py <config file>")
        sys.exit(1)
    config_filename = sys.argv[1]

    current_file = Path(__file__).resolve()
    script_dir = current_file.parent
    config_path = Path(script_dir / 'experiments')
    config_dict = ymlconfig.load_config(config_path / config_filename)
    config = ymlconfig.dict_to_namespace(config_dict)

    outputdir = (config_path / config.experiment.classification_type
                 / config.experiment.stage / config.experiment.tag)
    outputdir.mkdir(parents=True, exist_ok=True)

    shutil.copy(config_path / config_filename, outputdir / config_filename)

    consolidate_selection(config, config_path, outputdir)
    instance_table, _, _ = consolidate_counterfactuals(config, config_path, outputdir)
    plot_case_study_counterfactuals(config, config_path, outputdir)
    plot_remaining_counterfactuals(config, config_path, outputdir, instance_table)


if __name__ == "__main__":
    main()
