"""
Produce the exploratory data analysis (EDA) report for the binary DPN study.

Generates the two main-text tables, the two main-text figures, and the
Additional file 1 material for the BMC Medical Informatics and Decision Making
submission, together with a standalone caption for every artefact.

The analysis universe is the modelling feature set, i.e. every feature except
the groups listed under analysis.excluded_groups in the config (nerve conduction
studies, which enter the 2009 Toronto consensus definition of the outcome).
Nerve conduction variables are still summarised -- in the eligibility audit of
Table 2 and the descriptive table of Additional file 1 -- as the evidence for
their exclusion rather than as candidate predictors.

Every number quoted in a caption is computed here and interpolated, and the key
figures are additionally emitted as LaTeX macros in numbers.tex, so the
manuscript prose cannot drift from the tables.

Usage:
    python edareport.py bin_eda_final_202608.yml
"""
import matplotlib
matplotlib.use('Agg')

import json
import re
import shutil
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, norm, rankdata
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

from dataload import DPN_data
import ymlconfig

# ---------------------------------------------------------------------------
# Presentation constants
# ---------------------------------------------------------------------------

# Okabe-Ito, chosen for colour-vision deficiency safety. Every colour encoding in
# this report is duplicated by a shape or fill encoding so the figures survive
# greyscale printing.
GROUP_COLOURS = {
    "Profile":                  "#0072B2",
    "Comorbidity":              "#E69F00",
    "Neurological examination": "#009E73",
    "MNSI":                     "#CC79A7",
    "Sudoscan":                 "#56B4E9",
    "Nerve conduction study":   "#D55E00",
}
OUTCOME_COLOURS = {0: "#2166AC", 1: "#B2182B"}   # Unconfirmed / Confirmed
OUTCOME_LABELS = {0: "Unconfirmed", 1: "Confirmed"}
# Abbreviations for the narrow multi-panel grids, where the full labels collide.
SHORT_OUTCOME_LABELS = {0: "Unconf.", 1: "Conf."}

MM_PER_INCH = 25.4

# Units are reported in the tables so the descriptive statistics are
# interpretable without reference to the dataset documentation.
UNITS = {
    "AGE": "years", "DM_DUR": "years", "HBA1C": "%", "MNSI": "score",
    "FEET_MEAN_ESC": "uS", "HAND_MEAN_ESC": "uS",
    "FEET_PCT_ASYM": "%", "HAND_PCT_ASYM": "%", "NS": "score", "CAS": "%",
    "SSA_L": "uV", "SSA_R": "uV", "SPSA_L": "uV", "SPSA_R": "uV",
    "SSC_L": "m/s", "SSC_R": "m/s", "SPSC_L": "m/s", "SPSC_R": "m/s",
    "MCV_L": "m/s", "MCV_R": "m/s",
    "DL_L": "ms", "DL_R": "ms", "FWAVE_L": "ms", "FWAVE_R": "ms",
    "CMAPANK_L": "mV", "CMAPANK_R": "mV", "CMAPKNE_L": "mV", "CMAPKNE_R": "mV",
}

# The level counted as the event for each binary feature. Y/M map to 1 in the
# loader, so SEX == 1 is male.
BINARY_LEVEL = {"SEX": "male"}

# Nerve conduction variables where a recorded zero encodes an absent response
# ('NR' / 'NO F WAVE' in the source spreadsheet) rather than a measurement of
# zero. Flagged in the descriptive tables so a zero is never read as a value.
NON_RECORDABLE_ZERO_PREFIXES = ("SSA", "SSC", "SPSA", "SPSC", "MCV", "DL",
                                "CMAPANK", "CMAPKNE", "FWAVE")


def set_publication_style():
    """Apply a figure style suited to BMC's single-column, 170 mm page."""
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size":         8,
        "axes.titlesize":    9,
        "axes.titleweight":  "bold",
        "axes.labelsize":    8,
        "axes.linewidth":    0.7,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "xtick.major.size":  3,
        "ytick.major.size":  3,
        "legend.fontsize":   7,
        "legend.frameon":    False,
        "figure.dpi":        150,
        "savefig.dpi":       600,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
        "lines.linewidth":   1.0,
    })


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def auc_from_ranks(y, x):
    """Directional AUC = P(x higher in a random case than in a random control).

    Equivalent to sklearn's roc_auc_score (ties handled by mid-ranks) but built
    from ranks directly, which keeps the bootstrap below cheap enough to run for
    every feature. Values above 0.5 mean the feature is higher in the Confirmed
    group, values below 0.5 mean it is lower.
    """
    r = rankdata(x)
    n1 = int(np.sum(y == 1))
    n0 = int(np.sum(y == 0))
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def auc_bootstrap_ci(y, x, n_boot, rng, alpha=0.05):
    """Percentile bootstrap CI for the directional AUC, resampling within group.

    Stratified resampling keeps the case/control split of every replicate equal
    to the observed one, which matters here because the groups are of very
    unequal size (130 vs 57).
    """
    y = np.asarray(y)
    x = np.asarray(x, dtype=float)
    idx1 = np.flatnonzero(y == 1)
    idx0 = np.flatnonzero(y == 0)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        take = np.concatenate([rng.choice(idx1, idx1.size, replace=True),
                               rng.choice(idx0, idx0.size, replace=True)])
        stats[b] = auc_from_ranks(y[take], x[take])
    return tuple(np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def odds_ratio_ci(table, alpha=0.05):
    """Odds ratio with Wald CI, Haldane-Anscombe corrected when a cell is empty.

    `table` is [[control_no, control_yes], [case_no, case_yes]]. The 0.5
    correction keeps GBS (2 positives) and PAOD (8 positives) estimable instead
    of returning infinities.
    """
    t = np.asarray(table, dtype=float)
    corrected = (t == 0).any()
    if corrected:
        t = t + 0.5
    (a, b), (c, d) = t
    or_hat = (d * a) / (c * b)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z = norm.ppf(1 - alpha / 2)
    return or_hat, or_hat * np.exp(-z * se), or_hat * np.exp(z * se), corrected


def binary_test(table, expected_min):
    """Fisher's exact test when any expected cell is small, chi-square otherwise.

    Returns (p, test_name). Chi-square is unreliable for the rare comorbidities
    in this cohort, so the choice is made per feature from the expected counts
    rather than applied uniformly.
    """
    t = np.asarray(table, dtype=float)
    _, p_chi, _, expected = chi2_contingency(t, correction=False)
    if expected.min() < expected_min:
        _, p = fisher_exact(t)
        return p, "Fisher"
    return p_chi, "chi-square"


def wilson_ci(k, n, alpha=0.05):
    """Wilson score interval for a proportion; stable for small or extreme counts."""
    if n == 0:
        return np.nan, np.nan
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    # The Wilson interval always contains p; clamping against it removes the
    # rounding error that otherwise excludes p when k equals 0 or n.
    return min(max(0.0, centre - half), p), max(min(1.0, centre + half), p)


def benjamini_hochberg(pvals):
    """Benjamini-Hochberg false discovery rate q-values."""
    return multipletests(np.asarray(pvals, dtype=float), method="fdr_bh")[1]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def latex_escape(text):
    """Escape the characters that appear in this study's feature codes and units."""
    return (str(text).replace("\\", r"\textbackslash{}")
                     .replace("_", r"\_").replace("%", r"\%")
                     .replace("&", r"\&").replace("#", r"\#"))


def fmt_p(p):
    """Three-decimal p-value with the conventional '< 0.001' floor."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def decimals_for(values):
    """Use whole numbers for integer-valued features and one decimal otherwise."""
    v = np.asarray(values, dtype=float)
    return 0 if np.allclose(v, np.round(v)) else 1


def fmt_median_iqr(values, dec):
    if len(values) == 0:
        return "--"
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return f"{med:.{dec}f} [{q1:.{dec}f}--{q3:.{dec}f}]"


def fmt_n_pct(k, n):
    return f"{int(k)} ({100 * k / n:.1f})" if n else "--"


def fmt_ci(estimate, low, high, dec=2):
    return f"{estimate:.{dec}f} ({low:.{dec}f}--{high:.{dec}f})"


def write_latex_table(path, column_spec, header, body, caption, label, notes=()):
    """Write a booktabs table.

    `body` rows are either a list of already-formatted cells or the tuple
    ('GROUP', title), which renders a spanning group heading -- the layout BMC
    uses for characteristics tables organised by variable domain.
    """
    lines = [r"\begin{table}[!ht]", r"\centering", r"\footnotesize",
             r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\begin{tabular}{" + column_spec + "}", r"\toprule"]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    for row in body:
        if isinstance(row, tuple) and row and row[0] == "GROUP":
            lines.append(r"\addlinespace")
            lines.append(r"\multicolumn{" + str(len(header)) + r"}{l}{\textit{"
                         + latex_escape(row[1]) + r"}} \\")
        else:
            lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    for note in notes:
        lines.append(r"\begin{flushleft}\footnotesize " + note + r"\end{flushleft}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n")


def save_figure(fig, outputdir, stem, formats):
    """Save one figure in every configured format and return the written paths."""
    written = []
    for fmt in formats:
        path = outputdir / f"{stem}.{fmt}"
        fig.savefig(path)
        written.append(path)
    plt.close(fig)
    return written


def write_caption(outputdir, stem, caption):
    """Write the standalone caption that accompanies an artefact."""
    path = outputdir / f"{stem}_caption.txt"
    path.write_text(caption.strip() + "\n")
    return path


def half_violin(ax, values, position, colour, side=1, width=0.7):
    """Draw one half of a violin, the density half of a raincloud panel."""
    if len(values) < 2 or np.allclose(values, values[0]):
        return
    parts = ax.violinplot([values], positions=[position], widths=width,
                          showextrema=False, showmedians=False)
    for body in parts["bodies"]:
        verts = body.get_paths()[0].vertices
        centre = position
        # Clip the mirrored density to a single side so the raw points and the
        # box can occupy the other side without overplotting.
        if side > 0:
            verts[:, 0] = np.clip(verts[:, 0], centre, np.inf)
        else:
            verts[:, 0] = np.clip(verts[:, 0], -np.inf, centre)
        body.set_facecolor(colour)
        body.set_alpha(0.35)
        body.set_edgecolor(colour)
        body.set_linewidth(0.6)


def raincloud_panel(ax, df, feature, target, rng, dec=1):
    """Half violin + box + jittered raw points for one feature, by outcome group."""
    for pos, level in enumerate([0, 1]):
        vals = df.loc[df[target] == level, feature].to_numpy(dtype=float)
        colour = OUTCOME_COLOURS[level]
        half_violin(ax, vals, pos + 1, colour, side=1)
        bp = ax.boxplot([vals], positions=[pos + 1 - 0.16], widths=0.14,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="white", linewidth=1.2),
                        whiskerprops=dict(linewidth=0.6),
                        capprops=dict(linewidth=0.6))
        for box in bp["boxes"]:
            box.set_facecolor(colour)
            box.set_alpha(0.85)
            box.set_linewidth(0.5)
        jitter = rng.normal(0, 0.035, size=len(vals))
        ax.scatter(np.full(len(vals), pos + 1 - 0.34) + jitter, vals,
                   s=3, color=colour, alpha=0.45, linewidths=0, zorder=3)
    # Group content spans roughly pos-0.35 to pos+0.35, so centre the tick on it.
    ax.set_xticks([1 - 0.11, 2 - 0.11])
    ax.set_xticklabels([SHORT_OUTCOME_LABELS[0], SHORT_OUTCOME_LABELS[1]])
    ax.set_xlim(0.35, 2.75)
    unit = UNITS.get(feature)
    ax.set_ylabel(f"{feature} ({unit})" if unit else feature, fontsize=7.5)


# ---------------------------------------------------------------------------
# Feature inventory and per-feature statistics
# ---------------------------------------------------------------------------

def build_inventory(D, config):
    """Map every feature to its group, type, modelling role and counterfactual role."""
    groups = {
        "Profile":                  list(D.profile_cols),
        "Comorbidity":              list(D.comorbidity_cols),
        "Neurological examination": list(D.neuro_cols),
        "MNSI":                     list(D.mnsi_col),
        "Sudoscan":                 list(D.sudo_cols),
        "Nerve conduction study":   list(D.ncs_cols),
    }
    excluded_groups = set(config.analysis.excluded_groups)
    actionable = [c.strip() for c in config.cf_features.actionable.split(",")]

    rows = []
    for group, features in groups.items():
        for feature in features:
            modelled = group not in excluded_groups
            if not modelled:
                role = "Excluded (outcome-definitional)"
            elif feature in actionable:
                role = "Candidate predictor, counterfactual-actionable"
            else:
                role = "Candidate predictor, fixed"
            rows.append({
                "feature": feature,
                "group": group,
                "kind": "binary" if feature in D.categorical_cols else "continuous",
                "modelled": modelled,
                "actionable": feature in actionable,
                "role": role,
            })
    inventory = pd.DataFrame(rows)
    assert set(actionable).issubset(set(inventory.loc[inventory.modelled, "feature"])), \
        "every actionable feature must be a modelled feature"
    return inventory


def feature_statistics(df, inventory, target, config, rng):
    """Compute the univariable statistics that drive every table and figure.

    Discrimination is reported as the directional AUC for all features so that
    continuous and binary predictors sit on one scale (the scale the models are
    later reported on). Binary features additionally get an odds ratio, which is
    the form clinical readers expect in a characteristics table. False discovery
    rate correction is applied within the modelled set and, separately, within
    the excluded nerve conduction set -- the two are different families of
    hypotheses and pooling them would distort both.
    """
    y = df[target].to_numpy()
    records = []
    for row in inventory.itertuples():
        x = df[row.feature].to_numpy(dtype=float)
        auc = auc_from_ranks(y, x)
        lo, hi = auc_bootstrap_ci(y, x, config.analysis.bootstrap_n, rng)
        rec = {"feature": row.feature, "group": row.group, "kind": row.kind,
               "modelled": row.modelled, "actionable": row.actionable,
               "role": row.role, "auc": auc, "auc_lo": lo, "auc_hi": hi}

        if row.kind == "binary":
            table = [[int(((df[target] == 0) & (df[row.feature] == 0)).sum()),
                      int(((df[target] == 0) & (df[row.feature] == 1)).sum())],
                     [int(((df[target] == 1) & (df[row.feature] == 0)).sum()),
                      int(((df[target] == 1) & (df[row.feature] == 1)).sum())]]
            p, test = binary_test(table, config.analysis.fisher_expected_min)
            or_hat, or_lo, or_hi, corrected = odds_ratio_ci(table)
            # Named odds_ratio rather than 'or': DataFrame.itertuples cannot expose
            # an attribute whose name is a Python keyword.
            rec.update({"p": p, "test": test, "odds_ratio": or_hat, "or_lo": or_lo,
                        "or_hi": or_hi, "or_corrected": corrected})
        else:
            _, p = mannwhitneyu(x[y == 1], x[y == 0], alternative="two-sided")
            rec.update({"p": p, "test": "Mann-Whitney"})

        # Zeros encoding an absent nerve response, kept in every summary but
        # counted so a reader never mistakes one for a measured value.
        if row.feature.startswith(NON_RECORDABLE_ZERO_PREFIXES) and not row.modelled:
            rec["n_nonrecordable"] = int((df[row.feature] == 0).sum())
            rec["n_nonrecordable_unconf"] = int(((df[row.feature] == 0) & (df[target] == 0)).sum())
            rec["n_nonrecordable_conf"] = int(((df[row.feature] == 0) & (df[target] == 1)).sum())
        records.append(rec)

    stats = pd.DataFrame(records)
    stats["q"] = np.nan
    for modelled in (True, False):
        mask = stats.modelled == modelled
        stats.loc[mask, "q"] = benjamini_hochberg(stats.loc[mask, "p"].values)
    stats["strength"] = (stats["auc"] - 0.5).abs()
    return stats


def parse_cleaning_report(path):
    """Summarise the loader's cleaning report into provenance counts.

    dataload writes a per-cell audit trail; this collapses it to one row per
    (column, substitution) pair plus the list of excluded records, which is the
    granularity a reporting-checklist reviewer asks for.
    """
    replacements, dropped = {}, []
    for line in path.read_text().splitlines():
        drop = re.match(r"^row\s+(\d+)\s+NaN in:\s*(.+)$", line)
        if drop:
            # dataload documents patient CODE as the dataframe row position + 1.
            dropped.append((int(drop.group(1)) + 1, drop.group(2).strip()))
            continue
        rep = re.match(r"^row\s+(\d+)\s+(\S+)\s+(.+?)\s*->\s*(.+)$", line)
        if rep:
            key = (rep.group(2), rep.group(3).strip(), rep.group(4).strip())
            replacements[key] = replacements.get(key, 0) + 1
    return replacements, dropped


# ---------------------------------------------------------------------------
# Table 1 -- cohort characteristics
# ---------------------------------------------------------------------------

def table1_cohort_characteristics(df, stats, target, outputdir, ctx):
    """Characteristics of the analysed cohort, by outcome group."""
    n_all, n0, n1 = ctx["n_analysed"], ctx["n_unconfirmed"], ctx["n_confirmed"]
    modelled = stats[stats.modelled].copy()
    body, csv_rows = [], []

    for group in ["Profile", "Comorbidity", "Neurological examination", "MNSI", "Sudoscan"]:
        block = modelled[modelled.group == group]
        if block.empty:
            continue
        body.append(("GROUP", group))
        for row in block.itertuples():
            marker = r"$\diamond$" if row.actionable else ""
            values = df[row.feature].to_numpy(dtype=float)
            if row.kind == "binary":
                level = BINARY_LEVEL.get(row.feature, "yes")
                name = f"{latex_escape(row.feature)}, {level}"
                k_all = int((df[row.feature] == 1).sum())
                k0 = int(((df[row.feature] == 1) & (df[target] == 0)).sum())
                k1 = int(((df[row.feature] == 1) & (df[target] == 1)).sum())
                cells = [fmt_n_pct(k_all, n_all), fmt_n_pct(k0, n0), fmt_n_pct(k1, n1)]
                effect = "OR " + fmt_ci(row.odds_ratio, row.or_lo, row.or_hi)
            else:
                unit = UNITS.get(row.feature)
                name = latex_escape(row.feature) + (f", {latex_escape(unit)}" if unit else "")
                dec = decimals_for(values)
                cells = [fmt_median_iqr(values, dec),
                         fmt_median_iqr(df.loc[df[target] == 0, row.feature].to_numpy(float), dec),
                         fmt_median_iqr(df.loc[df[target] == 1, row.feature].to_numpy(float), dec)]
                effect = "AUC " + fmt_ci(row.auc, row.auc_lo, row.auc_hi)
            body.append([name + " " + marker] + cells + [effect, fmt_p(row.p), fmt_p(row.q)])
            csv_rows.append({"group": group, "feature": row.feature, "kind": row.kind,
                             "actionable": row.actionable, "overall": cells[0],
                             "unconfirmed": cells[1], "confirmed": cells[2],
                             "auc": row.auc, "auc_lo": row.auc_lo, "auc_hi": row.auc_hi,
                             "effect": effect.replace("$", ""), "test": row.test,
                             "p": row.p, "q": row.q})

    header = [r"\textbf{Characteristic}",
              r"\textbf{Overall} (n=" + str(n_all) + ")",
              r"\textbf{Unconfirmed} (n=" + str(n0) + ")",
              r"\textbf{Confirmed} (n=" + str(n1) + ")",
              r"\textbf{Effect size (95\% CI)}", r"\textbf{p}", r"\textbf{q}"]
    notes = [
        r"Continuous variables are median [interquartile range] and were compared with the "
        r"Mann--Whitney U test; binary variables are n (\%) of the stated level and were compared "
        r"with the chi-square test, or Fisher's exact test where any expected cell count was below "
        + str(ctx["fisher_expected_min"]) + r". "
        r"Effect size is the directional area under the receiver operating characteristic curve "
        r"(AUC) for continuous variables and the odds ratio (OR) for binary variables; an AUC above "
        r"0.5 indicates higher values in the Confirmed group and an AUC below 0.5 indicates lower "
        r"values. AUC confidence intervals are stratified percentile bootstrap intervals ("
        + str(ctx["bootstrap_n"]) + r" resamples). "
        r"q denotes the Benjamini--Hochberg false discovery rate across the "
        + str(ctx["n_modelled_features"]) + r" candidate predictors. "
        r"$\diamond$ marks the features the counterfactual engine is permitted to vary.",
    ]
    caption = (
        r"\textbf{Characteristics of the analysed cohort, by diabetic peripheral neuropathy status.} "
        r"Values are shown for all " + str(n_all) + r" analysed patients and separately for the "
        + str(n0) + r" Unconfirmed and " + str(n1) + r" Confirmed patients, for each of the "
        + str(ctx["n_modelled_features"]) + r" candidate predictors available to the models."
    )
    tex_path = outputdir / "table1_cohort_characteristics.tex"
    write_latex_table(tex_path, "l" + "c" * 3 + "c" + "cc", header, body,
                      caption, "tab:cohort", notes)
    csv_path = outputdir / "table1_cohort_characteristics.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    strongest = modelled.loc[modelled.strength.idxmax()]
    n_sig = int((modelled.q < 0.05).sum())
    act = modelled[modelled.actionable]
    act_sig = act[act.q < 0.05]
    caption_txt = f"""
Table 1 Characteristics of the analysed cohort, by diabetic peripheral neuropathy status.
Values are given for all {n_all} analysed patients and separately for the {n0} patients
classified as Unconfirmed and the {n1} classified as Confirmed under the 2009 Toronto
consensus criteria, for each of the {ctx['n_modelled_features']} candidate predictors
available to the machine learning models. Continuous variables are summarised as median
[interquartile range] and compared using the Mann-Whitney U test; binary variables are
summarised as n (%) of the stated level and compared using the chi-square test, or Fisher's
exact test where any expected cell count fell below {ctx['fisher_expected_min']}, as was the
case for the rarest comorbidities. Effect size is expressed as the directional area under the
receiver operating characteristic curve (AUC) for continuous variables, where a value above
0.5 indicates higher values among Confirmed patients and a value below 0.5 indicates lower
values, and as the odds ratio for binary variables; confidence intervals for the AUC are
stratified percentile bootstrap intervals based on {ctx['bootstrap_n']} resamples. The final
column reports Benjamini-Hochberg false discovery rate q-values computed across all
{ctx['n_modelled_features']} candidate predictors, of which {n_sig} reached q < 0.05. The
strongest single predictor was {strongest.feature} (AUC {strongest.auc:.3f}). Diamonds identify
the {len(act)} features that the counterfactual engine is permitted to modify; of these, only
{', '.join(act_sig.feature) if len(act_sig) else 'none'} reached q < 0.05, indicating that the
modifiable feature set carries limited univariable signal relative to the fixed predictors.
"""
    write_caption(outputdir, "table1_cohort_characteristics", caption_txt)
    return {"tex": tex_path, "csv": csv_path, "n_sig": n_sig,
            "strongest": strongest, "actionable_sig": list(act_sig.feature)}


# ---------------------------------------------------------------------------
# Table 2 -- feature eligibility and leakage audit
# ---------------------------------------------------------------------------

def table2_feature_eligibility(stats, outputdir, ctx):
    """Role of every feature group in the analysis, with the evidence for exclusion."""
    body, csv_rows = [], []
    order = ["Profile", "Comorbidity", "Neurological examination", "MNSI",
             "Sudoscan", "Nerve conduction study"]
    for group in order:
        block = stats[stats.group == group]
        if block.empty:
            continue
        body.append(("GROUP", group))
        for row in block.itertuples():
            if not row.modelled:
                role = "Excluded"
            elif row.actionable:
                role = r"Predictor, actionable $\diamond$"
            else:
                role = "Predictor, fixed"
            nr = ""
            if not np.isnan(getattr(row, "n_nonrecordable", np.nan)):
                nr = (f"{int(row.n_nonrecordable)} "
                      f"({100 * row.n_nonrecordable / ctx['n_analysed']:.1f})")
            body.append([latex_escape(row.feature), role,
                         fmt_ci(row.auc, row.auc_lo, row.auc_hi, dec=3),
                         fmt_p(row.q), nr if nr else "--"])
            csv_rows.append({"feature": row.feature, "group": group, "role": row.role,
                             "auc": row.auc, "auc_lo": row.auc_lo, "auc_hi": row.auc_hi,
                             "p": row.p, "q": row.q,
                             "n_nonrecordable": getattr(row, "n_nonrecordable", np.nan)})

    ncs = stats[~stats.modelled]
    header = [r"\textbf{Feature}", r"\textbf{Role}", r"\textbf{AUC (95\% CI)}",
              r"\textbf{q}", r"\textbf{Non-recordable, n (\%)}"]
    notes = [
        r"AUC is the univariable directional area under the receiver operating characteristic "
        r"curve for the feature used alone, with a stratified percentile bootstrap confidence "
        r"interval. q-values are Benjamini--Hochberg false discovery rates computed within the "
        + str(ctx["n_modelled_features"]) + r" candidate predictors and, separately, within the "
        + str(len(ncs)) + r" excluded nerve conduction variables. "
        r"Non-recordable counts the patients whose nerve response was absent "
        r"(recorded as \textit{NR} or \textit{NO F WAVE} in the source data and stored as zero); "
        r"these zeros are retained in all summaries. "
        r"$\diamond$ marks the counterfactual-actionable features.",
    ]
    caption = (r"\textbf{Feature eligibility and target-leakage audit.} Role of each feature in "
               r"the analysis together with its univariable discrimination, showing why nerve "
               r"conduction variables are withheld from the models.")
    tex_path = outputdir / "table2_feature_eligibility.tex"
    write_latex_table(tex_path, "llccc", header, body, caption, "tab:eligibility", notes)
    csv_path = outputdir / "table2_feature_eligibility.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    ncs_lo, ncs_hi = ncs.strength.add(0.5).min(), ncs.strength.add(0.5).max()
    modelled = stats[stats.modelled]
    best_modelled = modelled.loc[modelled.strength.idxmax()]
    caption_txt = f"""
Table 2 Feature eligibility and target-leakage audit.
The table states the role of each of the {len(stats)} recorded features and reports the
discrimination it achieves on its own, expressed as the directional area under the receiver
operating characteristic curve with a stratified bootstrap confidence interval. Under the 2009
Toronto consensus criteria, a classification of confirmed diabetic peripheral neuropathy
requires an abnormal nerve conduction study, so the {len(ncs)} nerve conduction variables form
part of the definition of the outcome rather than being independent of it. Their univariable
discrimination, ranging from {ncs_lo:.2f} to {ncs_hi:.2f} in absolute terms, is the expected
signature of this circularity and is reported here as the evidence for excluding them from
model development, not as a measure of predictive value; by comparison the strongest permitted
predictor, {best_modelled.feature}, reaches {best_modelled.auc:.3f}. The remaining
{ctx['n_modelled_features']} features constitute the modelling set, of which
{int(stats.actionable.sum())} are additionally designated as actionable and may be varied
during counterfactual generation. The final column reports patients whose nerve response was
absent, recorded as NR or NO F WAVE in the source data and stored as zero; these values are
retained unchanged in all analyses and are counted here so that a stored zero is not read as a
measured value. q-values are Benjamini-Hochberg false discovery rates computed within the
modelling set and, separately, within the excluded nerve conduction set.
"""
    write_caption(outputdir, "table2_feature_eligibility", caption_txt)
    return {"tex": tex_path, "csv": csv_path, "ncs_lo": ncs_lo, "ncs_hi": ncs_hi}


# ---------------------------------------------------------------------------
# Figure 1 -- cohort overview
# ---------------------------------------------------------------------------

def _text_width(ax, renderer, text, fontsize, fontweight="normal"):
    """Width of `text`, in ax data coordinates, as it would actually render."""
    artist = ax.text(0, 0, text, ha="center", va="center", fontsize=fontsize, fontweight=fontweight)
    bbox = artist.get_window_extent(renderer=renderer)
    (x0, _), (x1, _) = ax.transData.inverted().transform([(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)])
    artist.remove()
    return abs(x1 - x0)


def figure1a_participant_flow(ctx, outputdir, formats):
    """STROBE-style participant flow, drawn as boxes and arrows.

    The axes fill the whole figure (no margins), so one data unit is exactly
    one inch; box and canvas sizes can then be grown together, in lockstep,
    to fit measured label text without invalidating earlier measurements.
    """
    set_publication_style()
    xlim, ylim = 11.5, 10
    fig = plt.figure(figsize=(3.4, 5.1))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)
    ax.axis("off")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inches_per_unit = fig.get_figwidth() / xlim

    box_height = 1.7
    left = 1.4
    fontsize = 7
    boxes = [
        (8.6, f"Screened for eligibility\nn = {ctx['n_screened']}"),
        (5.9, f"Enrolled\nn = {ctx['n_enrolled']}"),
        (3.2, f"Analyzed\nn = {ctx['n_analysed']}"),
    ]
    padding = 0.7
    box_width = max(4.4, padding + max(_text_width(ax, renderer, text, fontsize)
                                       for _, text in boxes))
    centre = left + box_width / 2
    right = left + box_width

    excl = [
        (7.25, f"Excluded n = {ctx['n_screened'] - ctx['n_enrolled']}\n"
               + textwrap.fill(ctx["screening_exclusion_reason"], 22)),
        (4.55, f"Excluded n = {ctx['n_enrolled'] - ctx['n_analysed']}\nincomplete measurements\n"
               f"(patient codes {ctx['dropped_codes_text']})"),
    ]
    excl_gap = 0.4
    excl_fontsize = 6
    excl_width = max(_text_width(ax, renderer, text, excl_fontsize) for _, text in excl)
    content_right = right + excl_gap + 0.2 + excl_width
    new_xlim = content_right + left  # mirror the left margin so both sides match
    if new_xlim != xlim:
        fig.set_figwidth(fig.get_figwidth() + (new_xlim - xlim) * inches_per_unit)
        xlim = new_xlim
        ax.set_xlim(0, xlim)

    shift = 0.5 * left  # nudge the whole drawing right within the same canvas width
    ax.set_xlim(-shift, xlim - shift)

    for y, text in boxes:
        ax.add_patch(Rectangle((left, y - box_height / 2), box_width, box_height,
                               facecolor="#F2F2F2", edgecolor="0.35", linewidth=0.7))
        ax.text(centre, y, text, ha="center", va="center", fontsize=fontsize)
    for y_from, y_to in [(8.6, 5.9), (5.9, 3.2)]:
        ax.add_patch(FancyArrowPatch((centre, y_from - box_height / 2), (centre, y_to + box_height / 2),
                                     arrowstyle="-|>", mutation_scale=8,
                                     linewidth=0.7, color="0.35"))
    for y, text in excl:
        ax.add_patch(FancyArrowPatch((centre, y), (right + excl_gap, y), arrowstyle="-|>",
                                     shrinkA=0, mutation_scale=8, linewidth=0.7, color="0.35"))
        ax.text(right + excl_gap + 0.2, y, text, ha="left", va="center", fontsize=excl_fontsize, color="0.25")

    split_texts = [f"Confirmed\nn = {ctx['n_confirmed']}", f"Unconfirmed\nn = {ctx['n_unconfirmed']}"]
    split_fontsize = 6.5
    split_box_width = 0.5 + max(_text_width(ax, renderer, t, split_fontsize, fontweight="bold")
                                for t in split_texts)
    split_box_height = 1.1
    split_gap = 0.5
    split_y = 1.15
    branch_y = 1.95
    stem_top = 3.2 - box_height / 2
    left_x = centre - split_gap / 2 - split_box_width / 2
    right_x = centre + split_gap / 2 + split_box_width / 2

    ax.add_patch(FancyArrowPatch((centre, stem_top), (centre, branch_y), arrowstyle="-",
                                 shrinkB=0, linewidth=0.7, color="0.35"))
    ax.add_patch(FancyArrowPatch((left_x, branch_y), (right_x, branch_y), arrowstyle="-",
                                 shrinkA=0, shrinkB=0, linewidth=0.7, color="0.35"))
    for x, text in zip((left_x, right_x), split_texts):
        ax.add_patch(FancyArrowPatch((x, branch_y), (x, split_y + split_box_height / 2), arrowstyle="-|>",
                                     shrinkA=0, mutation_scale=8, linewidth=0.7, color="0.35"))
        ax.add_patch(Rectangle((x - split_box_width / 2, split_y - split_box_height / 2),
                               split_box_width, split_box_height,
                               facecolor="#F2F2F2", edgecolor="0.35", linewidth=0.7))
        ax.text(x, split_y, text, ha="center", va="center", ma="center",
               fontsize=split_fontsize, fontweight="bold")

    # svg.fonttype "none" keeps panel labels as real, editable <text> elements
    # in the SVG rather than flattening them to glyph outlines.
    all_formats = list(dict.fromkeys([*formats, "svg"]))
    with plt.rc_context({"svg.fonttype": "none"}):
        paths = save_figure(fig, outputdir, "fig1a_participant_flow", all_formats)

    caption_txt = f"""
Fig. 1a Participant flow. Of {ctx['n_screened']} patients screened, {ctx['n_enrolled']} were
enrolled after applying the exclusion criteria, and {ctx['n_analysed']} were analyzed after
{ctx['n_enrolled'] - ctx['n_analysed']} records (patient codes {ctx['dropped_codes_text']}) were
removed for incomplete measurements.
"""
    write_caption(outputdir, "fig1a_participant_flow", caption_txt)
    return {"paths": paths}


CLASSIFICATION_COMPOSITION_CAPTION = """
Fig. 1b Classification composition. {n_confirmed} patients
({prevalence:.1f}%) met the 2009 Toronto consensus criteria for confirmed diabetic
peripheral neuropathy and {n_unconfirmed} did not; the latter group combines the
negative, possible and probable categories. The cohort therefore provides {epv:.1f}
minority-class events per candidate predictor across the {n_modelled_features} features
available to the models, a low ratio that motivates the repeated cross-validation strategy used
throughout and that should temper the interpretation of any single fitted coefficient.
"""


def figure1bh_classification_composition(ctx, outputdir, formats):
    """Outcome balance, horizontal bars, with the prevalence/EPV annotation."""
    set_publication_style()
    fig, ax = plt.subplots(figsize=(3.6, 2.3))
    counts = [ctx["n_unconfirmed"], ctx["n_confirmed"]]
    bars = ax.barh([0, 1], counts, height=0.42,
                   color=[OUTCOME_COLOURS[0], OUTCOME_COLOURS[1]], alpha=0.85)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                f"{count} ({100 * count / ctx['n_analysed']:.1f}%)",
                va="center", fontsize=7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([OUTCOME_LABELS[0], OUTCOME_LABELS[1]])
    ax.set_xlabel("Patients")
    ax.set_xlim(0, max(counts) * 1.38)
    ax.set_ylim(-0.75, 1.55)
    ax.text(0.98, 0.05,
            f"prevalence {100 * ctx['prevalence']:.1f}%\n"
            f"{ctx['n_modelled_features']} predictors\n"
            f"{ctx['epv']:.1f} events per variable",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color="0.25")
    paths = save_figure(fig, outputdir, "fig1bh_classification_composition", formats)

    caption_txt = CLASSIFICATION_COMPOSITION_CAPTION.format(
        n_confirmed=ctx["n_confirmed"], prevalence=100 * ctx["prevalence"],
        n_unconfirmed=ctx["n_unconfirmed"], epv=ctx["epv"],
        n_modelled_features=ctx["n_modelled_features"])
    write_caption(outputdir, "fig1bh_classification_composition", caption_txt)
    return {"paths": paths}


def figure1bv_classification_composition(ctx, outputdir, formats):
    """Outcome balance, vertical bars, sized to sit side by side with Figure 1a."""
    set_publication_style()
    fig, ax = plt.subplots(figsize=(2.6, 6.2))
    counts = [ctx["n_unconfirmed"], ctx["n_confirmed"]]
    bars = ax.bar([0, 1], counts, width=0.55,
                  color=[OUTCOME_COLOURS[0], OUTCOME_COLOURS[1]], alpha=0.85)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02,
                f"{count}\n({100 * count / ctx['n_analysed']:.1f}%)",
                ha="center", va="bottom", fontsize=7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([OUTCOME_LABELS[0], OUTCOME_LABELS[1]])
    ax.set_ylabel("Patients")
    ax.set_ylim(0, max(counts) * 1.42)
    ax.set_xlim(-0.85, 1.85)
    ax.text(0.5, 0.98,
            f"prevalence {100 * ctx['prevalence']:.1f}%\n"
            f"{ctx['n_modelled_features']} predictors\n"
            f"{ctx['epv']:.1f} events per variable",
            transform=ax.transAxes, ha="center", va="top", fontsize=6, color="0.25")
    paths = save_figure(fig, outputdir, "fig1bv_classification_composition", formats)

    caption_txt = CLASSIFICATION_COMPOSITION_CAPTION.format(
        n_confirmed=ctx["n_confirmed"], prevalence=100 * ctx["prevalence"],
        n_unconfirmed=ctx["n_unconfirmed"], epv=ctx["epv"],
        n_modelled_features=ctx["n_modelled_features"])
    write_caption(outputdir, "fig1bv_classification_composition", caption_txt)
    return {"paths": paths}


# ---------------------------------------------------------------------------
# Figure 2 -- univariable discrimination
# ---------------------------------------------------------------------------

def figure2_univariable_discrimination(stats, outputdir, ctx, formats):
    """Forest of per-feature AUC, ordered by discrimination strength."""
    set_publication_style()
    modelled = stats[stats.modelled].sort_values("strength")
    width = ctx["figure_width_mm"] / MM_PER_INCH
    fig, ax = plt.subplots(figsize=(width * 0.72, 0.21 * len(modelled) + 1.5))

    ypos = np.arange(len(modelled))
    for y, row in zip(ypos, modelled.itertuples()):
        colour = GROUP_COLOURS[row.group]
        ax.plot([row.auc_lo, row.auc_hi], [y, y], color=colour, linewidth=1.1,
                solid_capstyle="round", zorder=2)
        # Shape encodes counterfactual actionability and fill encodes
        # significance, so neither meaning depends on colour alone.
        ax.scatter([row.auc], [y], marker="D" if row.actionable else "o",
                   s=30 if row.actionable else 22,
                   facecolor=colour if row.q < 0.05 else "white",
                   edgecolor=colour, linewidths=0.9, zorder=3)

    ax.axvline(0.5, color="0.35", linestyle="--", linewidth=0.8)
    ax.axvline(ctx["model_auc"], color="0.15", linestyle=":", linewidth=1.0)
    # Annotations sit in headroom above the top row so they never overlap a marker.
    ax.text(ctx["model_auc"], len(modelled) + 1.5,
            f" {ctx['model_name']} model\n {ctx['model_auc']:.3f}",
            fontsize=6.5, va="top", ha="left", color="0.35")
    ax.text(0.5, len(modelled) + 1.5, "no \ndiscrimination ", fontsize=6.5,
            va="top", ha="right", color="0.35")

    ax.set_yticks(ypos)
    ax.set_yticklabels(modelled.feature, fontsize=7)
    # Headroom above the top row for the reference annotations, and below the
    # bottom row for the legend, so neither can cover an interval.
    ax.set_ylim(-2.6, len(modelled) + 1.7)
    ax.set_xlim(0.15, 1.0)

    handles = [Line2D([], [], marker="o", linestyle="", markerfacecolor=c,
                      markeredgecolor=c, markersize=5, label=g)
               for g, c in GROUP_COLOURS.items() if g in set(modelled.group)]
    handles += [
        Line2D([], [], marker="D", linestyle="", markerfacecolor="none",
               markeredgecolor="0.4", markersize=5, label="counterfactual-actionable"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=6.2, ncol=1,
              handletextpad=0.4, borderaxespad=0.4)
    fig.tight_layout()
    paths = save_figure(fig, outputdir, "fig2_univariable_discrimination", formats)

    best = modelled.loc[modelled.strength.idxmax()]
    act = modelled[modelled.actionable]
    act_lo, act_hi = act.auc.min(), act.auc.max()
    act_sig = act[act.q < 0.05]
    inverted = modelled[modelled.auc < 0.5]
    caption_txt = f"""
Fig. 2 Univariable discrimination of each candidate predictor.
Each row shows the area under the receiver operating characteristic curve obtained when a
single feature is used alone to separate Confirmed from Unconfirmed patients, with a stratified
percentile bootstrap 95% confidence interval based on {ctx['bootstrap_n']} resamples; features
are ordered by discrimination strength. A value of 0.5, marked by the dashed line, indicates no
discrimination, while values below 0.5 indicate features whose values are lower among Confirmed
patients ({', '.join(inverted.feature)}). Marker colour denotes the clinical domain and diamonds
denote the {len(act)} features the counterfactual engine is permitted to vary; circles denote the
remaining candidate predictors. Regardless of shape, a filled marker denotes a feature reaching a
Benjamini-Hochberg false discovery rate below 0.05, while an open marker denotes one that does
not. The
dotted line marks the discrimination achieved by the tuned {ctx['model_name']} model
({ctx['model_auc']:.3f}), placing the multivariable result on the same scale as its inputs: the
strongest individual feature, {best.feature}, already reaches {best.auc:.3f}, so the modelling
gain over a single bedside measurement is modest. The actionable features span
{act_lo:.3f} to {act_hi:.3f} and cluster at the no-discrimination line, with only
{', '.join(act_sig.feature) if len(act_sig) else 'none'} reaching q < 0.05. Counterfactual
explanations in this study are therefore constrained to move the features carrying the least
marginal signal, which is the expected origin of the large feature displacements those
explanations require.
"""
    write_caption(outputdir, "fig2_univariable_discrimination", caption_txt)
    return {"paths": paths, "best": best, "act_lo": act_lo, "act_hi": act_hi}


# ---------------------------------------------------------------------------
# Additional file 1
# ---------------------------------------------------------------------------

def s1_continuous_rainclouds(df, stats, target, outputdir, ctx, formats, rng):
    set_publication_style()
    features = stats[(stats.modelled) & (stats.kind == "continuous")].sort_values(
        "strength", ascending=False).feature.tolist()
    ncols = 5
    nrows = int(np.ceil(len(features) / ncols))
    width = ctx["figure_width_mm"] / MM_PER_INCH
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, 2.4 * nrows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, feature in zip(axes, features):
        raincloud_panel(ax, df, feature, target, rng)
        row = stats[stats.feature == feature].iloc[0]
        ax.set_title(f"{feature}\nAUC {row.auc:.2f}, q {fmt_p(row.q).replace('$<$', '<')}",
                     fontsize=6.8, fontweight="normal", color="0.3")
    for ax in axes[len(features):]:
        ax.set_visible(False)
    paths = save_figure(fig, outputdir, "s1_continuous_rainclouds", formats)

    caption_txt = f"""
Additional file 1: Figure S1 Distribution of the continuous candidate predictors by outcome
group. Each panel shows, for one of the {len(features)} continuous features in the modelling
set, the kernel density of the values (right), a box plot of the median and interquartile range
(centre) and the individual patient observations (left), separately for the
{ctx['n_unconfirmed']} Unconfirmed and {ctx['n_confirmed']} Confirmed patients. Panels are
ordered by univariable discrimination and annotated with the area under the receiver operating
characteristic curve and the Benjamini-Hochberg q-value reported in Table 1. Showing the raw
observations alongside the summaries makes the degree of overlap between the groups visible:
even for the best-separating features the two distributions overlap substantially, which is
consistent with the moderate specificity of the fitted models and with the wide range of
predicted probabilities near the decision threshold that motivated the selection of borderline
cases for counterfactual analysis.
"""
    write_caption(outputdir, "s1_continuous_rainclouds", caption_txt)
    return {"paths": paths}


def s2_categorical_features(df, stats, target, outputdir, ctx, formats):
    set_publication_style()
    binary = stats[(stats.modelled) & (stats.kind == "binary")].sort_values("strength")
    width = ctx["figure_width_mm"] / MM_PER_INCH
    fig, (ax_p, ax_or) = plt.subplots(1, 2, figsize=(width, 0.30 * len(binary) + 1.6),
                                      gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.45})

    ypos = np.arange(len(binary))
    for y, row in zip(ypos, binary.itertuples()):
        for level, offset in [(0, -0.18), (1, 0.18)]:
            sub = df[df[target] == level]
            k, n = int((sub[row.feature] == 1).sum()), len(sub)
            lo, hi = wilson_ci(k, n)
            ax_p.plot([100 * lo, 100 * hi], [y + offset] * 2,
                      color=OUTCOME_COLOURS[level], linewidth=1.0)
            ax_p.scatter([100 * k / n], [y + offset], s=16,
                         color=OUTCOME_COLOURS[level], zorder=3)
    ax_p.set_yticks(ypos)
    ax_p.set_yticklabels(binary.feature, fontsize=7)
    ax_p.set_xlabel("Patients with the feature present (%)")
    ax_p.set_xlim(-2, 102)
    ax_p.legend(handles=[Line2D([], [], marker="o", linestyle="", color=OUTCOME_COLOURS[k],
                                markersize=5, label=v) for k, v in OUTCOME_LABELS.items()],
                loc="lower right", fontsize=6.5)
    ax_p.set_title("a  Prevalence with 95% CI", loc="left")

    for y, row in zip(ypos, binary.itertuples()):
        colour = GROUP_COLOURS[row.group]
        ax_or.plot([row.or_lo, row.or_hi], [y, y], color=colour, linewidth=1.1)
        ax_or.scatter([row.odds_ratio], [y], s=24, zorder=3,
                      facecolor=colour if row.q < 0.05 else "white",
                      edgecolor=colour, linewidths=0.9,
                      marker="D" if row.actionable else "o")
    ax_or.axvline(1, color="0.35", linestyle="--", linewidth=0.8)
    ax_or.set_xscale("log")
    ax_or.set_yticks(ypos)
    ax_or.set_yticklabels([])
    ax_or.set_xlabel("Odds ratio (log scale)")
    ax_or.set_title("b  Association with Confirmed status", loc="left")
    paths = save_figure(fig, outputdir, "s2_categorical_features", formats)

    sig = binary[binary.q < 0.05]
    caption_txt = f"""
Additional file 1: Figure S2 Binary candidate predictors and their association with confirmed
diabetic peripheral neuropathy. a Prevalence of each of the {len(binary)} binary features in the
Unconfirmed and Confirmed groups, with Wilson 95% confidence intervals, which remain valid for
the rare comorbidities where normal-approximation intervals do not. b Odds ratios for the same
features on a logarithmic scale, with Wald confidence intervals and a Haldane-Anscombe
correction applied where a cell count was zero; the dashed line marks the null value of 1.
Filled markers denote a Benjamini-Hochberg q-value below 0.05 and diamonds denote the features
available to the counterfactual engine. The neurological examination signs
({', '.join(sig[sig.group == 'Neurological examination'].feature)}) show the strongest and most
consistent associations, whereas the comorbidities cluster tightly around the null with wide
intervals reflecting their low prevalence, most extremely for GBS and PAOD. This pattern
explains why the comorbidity ablation had little effect on model performance despite
comorbidities forming the majority of the modifiable feature set.
"""
    write_caption(outputdir, "s2_categorical_features", caption_txt)
    return {"paths": paths}


def s3_redundancy(df, stats, outputdir, ctx, formats):
    """Correlation structure, the collinear pair, and variance inflation factors."""
    set_publication_style()
    features = stats[stats.modelled].feature.tolist()
    corr = df[features].corr(method="spearman")

    # Order the matrix by hierarchical clustering on correlation distance so that
    # mutually redundant features sit adjacent instead of in data-dictionary order.
    dist = 1 - corr.abs().to_numpy()
    np.fill_diagonal(dist, 0.0)
    link = hierarchy.linkage(squareform(dist, checks=False), method="average")
    order = [features[i] for i in hierarchy.leaves_list(link)]
    corr = corr.loc[order, order]

    X = df[features].astype(float).assign(const=1.0)
    vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(len(features))],
                    index=features).sort_values(ascending=False)
    high_vif = vif[vif > ctx["vif_threshold"]]

    width = ctx["figure_width_mm"] / MM_PER_INCH
    fig = plt.figure(figsize=(width, width * 1.05))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.55, 1.0], hspace=0.62, wspace=0.35, figure=fig)

    ax_h = fig.add_subplot(gs[0, :])
    im = ax_h.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax_h.set_xticks(range(len(order)))
    ax_h.set_xticklabels(order, rotation=90, fontsize=6)
    ax_h.set_yticks(range(len(order)))
    ax_h.set_yticklabels(order, fontsize=6)
    for tick, feature in zip(ax_h.get_yticklabels(), order):
        tick.set_color(GROUP_COLOURS[stats.loc[stats.feature == feature, "group"].iloc[0]])
    fig.colorbar(im, ax=ax_h, shrink=0.65, label="Spearman $\\rho$")
    ax_h.set_title("a  Correlation structure (clustered)", loc="left")

    # The most collinear pair, shown directly because it drives the VIF result.
    pair = ctx["top_pair"]
    ax_s = fig.add_subplot(gs[1, 0])
    ax_s.scatter(df[pair[0]], df[pair[1]], s=8, alpha=0.5, linewidths=0, color="0.35")
    ax_s.set_xlabel(pair[0])
    ax_s.set_ylabel(pair[1])
    ax_s.set_title(f"b  {pair[0]} vs {pair[1]} ($\\rho$ = {pair[2]:.2f})", loc="left")

    ax_v = fig.add_subplot(gs[1, 1])
    top = vif.head(8)[::-1]
    ax_v.barh(range(len(top)), top.values, height=0.6,
              color=["#B2182B" if v > ctx["vif_threshold"] else "0.6" for v in top.values])
    ax_v.axvline(ctx["vif_threshold"], color="0.25", linestyle="--", linewidth=0.8)
    ax_v.set_yticks(range(len(top)))
    ax_v.set_yticklabels(top.index, fontsize=6.5)
    ax_v.set_xlabel("Variance inflation factor")
    ax_v.set_title("c  Multicollinearity", loc="left")
    paths = save_figure(fig, outputdir, "s3_redundancy", formats)

    vif_csv = outputdir / "s3_variance_inflation_factors.csv"
    vif.rename("vif").to_frame().to_csv(vif_csv, index_label="feature")

    caption_txt = f"""
Additional file 1: Figure S3 Redundancy among the candidate predictors.
a Spearman correlation matrix of the {len(features)} modelling features, ordered by hierarchical
clustering on correlation distance so that mutually redundant features appear adjacent; tick
label colours denote the clinical domain. b The most strongly correlated pair,
{pair[0]} and {pair[1]} (rho = {pair[2]:.2f}). The Sudoscan neuropathy score is age-adjusted by
construction, so the two variables carry overlapping but not identical information. c Variance
inflation factors, with the threshold of {ctx['vif_threshold']} used for the collinearity-pruned
feature set marked by the dashed line. Exactly {len(high_vif)} features exceed that threshold
({', '.join(f'{k} = {v:.1f}' for k, v in high_vif.items())}), and both belong to the pair shown
in panel b. Removing them therefore discards two of the more informative predictors rather than
redundant noise, which accounts for the collinearity-pruned feature set performing no better
than the full set in the ablation experiments.
"""
    write_caption(outputdir, "s3_redundancy", caption_txt)
    return {"paths": paths, "vif": vif, "high_vif": high_vif, "csv": vif_csv}


def s4_data_provenance(replacements, dropped, outputdir, ctx):
    """Every alteration made between the source spreadsheet and the analysed data."""
    body, csv_rows = [], []
    by_column = {}
    for (column, old, new), count in replacements.items():
        by_column.setdefault((column, old, new), 0)
        by_column[(column, old, new)] += count

    def arrow(old, new):
        """Render a substitution in maths mode; '>' is unreliable in LaTeX text mode."""
        return (r"\textit{" + latex_escape(old.strip("'")) + r"} $\rightarrow$ "
                + latex_escape(new.strip("'")))

    body.append(("GROUP", "Value recoding"))
    for (column, old, new), count in sorted(by_column.items(), key=lambda kv: (-kv[1], kv[0])):
        body.append([latex_escape(column), arrow(old, new), str(count)])
        csv_rows.append({"category": "value recoding", "item": column,
                         "detail": f"{old} -> {new}", "n": count})
    body.append([r"\textit{all binary features}",
                 r"\textit{Y}/\textit{M} $\rightarrow$ 1, \textit{N}/\textit{F} $\rightarrow$ 0",
                 "deterministic"])
    csv_rows.append({"category": "value recoding", "item": "all binary features",
                     "detail": "Y/M -> 1, N/F -> 0", "n": "deterministic"})

    body.append(("GROUP", "Excluded records"))
    for code, columns in dropped:
        body.append([f"Patient {code}", latex_escape(f"missing: {columns}"), "1"])
        csv_rows.append({"category": "excluded record", "item": f"patient {code}",
                         "detail": f"missing: {columns}", "n": 1})

    header = [r"\textbf{Item}", r"\textbf{Change applied}", r"\textbf{Cells / records}"]
    notes = [r"Generated from the audit trail written by the data loader, so the table cannot "
             r"drift from the data actually analysed."]
    caption = (r"\textbf{Data provenance and completeness.} Every alteration applied between the "
               r"source spreadsheet and the analysed dataset.")
    tex_path = outputdir / "s4_data_provenance.tex"
    write_latex_table(tex_path, "llc", header, body, caption, "tab:provenance", notes)
    csv_path = outputdir / "s4_data_provenance.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    n_cells = sum(by_column.values())
    caption_txt = f"""
Additional file 1: Table S1 Data provenance and completeness.
The table records every alteration applied between the source spreadsheet and the analysed
dataset, generated directly from the audit trail written by the data loader so that it cannot
drift from the data actually analysed. A total of {n_cells} cells were recoded. Diabetes
duration values recorded as open-ended categories were converted to numeric values, and nerve
conduction entries recorded as NR or NO F WAVE, denoting an absent response, were stored as
zero; the latter are retained as zeros in all analyses and are counted per variable in Table 2.
Binary features recorded as Y/M and N/F were mapped deterministically to 1 and 0.
{len(dropped)} of the {ctx['n_enrolled']} enrolled records were excluded because one or more
numeric measurements were absent, leaving {ctx['n_analysed']} patients with complete data on all
recorded variables; no imputation was performed and no missing values remain in the analysed
dataset.
"""
    write_caption(outputdir, "s4_data_provenance", caption_txt)
    return {"tex": tex_path, "csv": csv_path, "n_cells": n_cells}


def s5_actionable_features(df, stats, target, outputdir, ctx, formats, rng):
    """Detail on the features the counterfactual engine may vary."""
    set_publication_style()
    act = stats[stats.actionable].sort_values("strength", ascending=False)
    width = ctx["figure_width_mm"] / MM_PER_INCH
    ncols = 3
    nrows = int(np.ceil(len(act) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, 2.5 * nrows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, row in zip(axes, act.itertuples()):
        if row.kind == "continuous":
            raincloud_panel(ax, df, row.feature, target, rng)
        else:
            for level in (0, 1):
                sub = df[df[target] == level]
                k, n = int((sub[row.feature] == 1).sum()), len(sub)
                lo, hi = wilson_ci(k, n)
                ax.bar([level], [100 * k / n], width=0.5, color=OUTCOME_COLOURS[level],
                       alpha=0.85)
                ax.errorbar([level], [100 * k / n],
                            yerr=[[100 * (k / n - lo)], [100 * (hi - k / n)]],
                            fmt="none", ecolor="0.3", linewidth=0.8, capsize=3)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([SHORT_OUTCOME_LABELS[0], SHORT_OUTCOME_LABELS[1]])
            ax.set_ylabel(f"{row.feature} present (%)", fontsize=7.5)
            ax.set_ylim(0, 100)
        ax.set_title(f"{row.feature}\nAUC {row.auc:.3f}, q {fmt_p(row.q).replace('$<$', '<')}",
                     fontsize=6.8, fontweight="normal", color="0.3")
    for ax in axes[len(act):]:
        ax.set_visible(False)
    paths = save_figure(fig, outputdir, "s5_actionable_features", formats)

    sig = act[act.q < 0.05]
    caption_txt = f"""
Additional file 1: Figure S4 The counterfactual-actionable features.
Distribution by outcome group of the {len(act)} features that the counterfactual engine is
permitted to vary, comprising the modifiable treatment and comorbidity variables; continuous
features are shown as density, box and raw-observation displays and binary features as
prevalence with Wilson 95% confidence intervals. Each panel is annotated with the univariable
area under the receiver operating characteristic curve and the Benjamini-Hochberg q-value from
Table 1. The distributions overlap almost completely: discrimination ranges from
{act.auc.min():.3f} to {act.auc.max():.3f}, and only
{', '.join(f'{r.feature} (q = {r.q:.3f})' for r in sig.itertuples()) if len(sig) else 'none'}
{'reaches' if len(sig) == 1 else 'reach'} a false discovery rate below 0.05. Because these are
the only features a counterfactual may change, the explanations generated for a given patient
must displace variables that individually carry little marginal information, which accounts for
the magnitude of the changes those explanations propose. This also yields a coherence check on
the counterfactual results: the actionable feature with genuine univariable signal would be
expected to appear among the most frequently altered features in the generated counterfactuals.
"""
    write_caption(outputdir, "s5_actionable_features", caption_txt)
    return {"paths": paths}


def s6_ncs_descriptives(df, stats, target, outputdir, ctx):
    """Descriptive summary of the excluded nerve conduction variables."""
    ncs = stats[~stats.modelled]
    body, csv_rows = [], []
    n0, n1 = ctx["n_unconfirmed"], ctx["n_confirmed"]
    for row in ncs.itertuples():
        values = df[row.feature].to_numpy(dtype=float)
        dec = decimals_for(values)
        unit = UNITS.get(row.feature)
        name = latex_escape(row.feature) + (f", {latex_escape(unit)}" if unit else "")
        nr0 = int(row.n_nonrecordable_unconf)
        nr1 = int(row.n_nonrecordable_conf)
        body.append([
            name,
            fmt_median_iqr(values, dec),
            fmt_median_iqr(df.loc[df[target] == 0, row.feature].to_numpy(float), dec),
            fmt_median_iqr(df.loc[df[target] == 1, row.feature].to_numpy(float), dec),
            f"{fmt_n_pct(nr0, n0)} / {fmt_n_pct(nr1, n1)}",
            fmt_ci(row.auc, row.auc_lo, row.auc_hi, dec=3),
            fmt_p(row.q),
        ])
        csv_rows.append({"feature": row.feature, "unit": unit,
                         "nonrecordable_unconfirmed": nr0, "nonrecordable_confirmed": nr1,
                         "auc": row.auc, "auc_lo": row.auc_lo, "auc_hi": row.auc_hi,
                         "p": row.p, "q": row.q})

    header = [r"\textbf{Variable}", r"\textbf{Overall}",
              r"\textbf{Unconfirmed} (n=" + str(n0) + ")",
              r"\textbf{Confirmed} (n=" + str(n1) + ")",
              r"\textbf{Non-recordable}", r"\textbf{AUC (95\% CI)}", r"\textbf{q}"]
    notes = [r"Values are median [interquartile range]. Non-recordable gives the count and "
             r"percentage of absent responses, Unconfirmed / Confirmed. These variables were "
             r"withheld from model development; see Table~\ref{tab:eligibility}."]
    caption = (r"\textbf{Nerve conduction study variables.} Descriptive summary of the variables "
               r"withheld from model development because they enter the definition of the outcome.")
    tex_path = outputdir / "s6_ncs_descriptives.tex"
    write_latex_table(tex_path, "lcccccc", header, body, caption, "tab:ncs", notes)
    csv_path = outputdir / "s6_ncs_descriptives.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    worst = ncs.loc[ncs.n_nonrecordable.idxmax()]
    caption_txt = f"""
Additional file 1: Table S2 Nerve conduction study variables.
Descriptive summary of the {len(ncs)} nerve conduction variables, which were recorded for every
analysed patient but withheld from model development because an abnormal nerve conduction study
forms part of the 2009 Toronto consensus definition of confirmed diabetic peripheral neuropathy.
Values are median [interquartile range] overall and by outcome group, alongside the count of
absent responses and the univariable area under the receiver operating characteristic curve.
Absent responses are frequent and strongly patterned by outcome: {worst.feature} was
non-recordable in {int(worst.n_nonrecordable_conf)} of {n1} Confirmed patients
({100 * worst.n_nonrecordable_conf / n1:.1f}%) compared with
{int(worst.n_nonrecordable_unconf)} of {n0} Unconfirmed patients
({100 * worst.n_nonrecordable_unconf / n0:.1f}%). These variables are summarised here for
completeness, since the dataset is newly collected and is being released alongside this work,
and to document the discrimination that motivated their exclusion. q-values are
Benjamini-Hochberg false discovery rates computed within this set of {len(ncs)} variables.
"""
    write_caption(outputdir, "s6_ncs_descriptives", caption_txt)
    return {"tex": tex_path, "csv": csv_path}


# ---------------------------------------------------------------------------
# Cross-artefact outputs
# ---------------------------------------------------------------------------

def write_numbers(outputdir, ctx, stats, extras):
    """Emit the headline numbers as LaTeX macros for use in the manuscript prose.

    Anything the text asserts about the cohort should be pulled from here rather
    than typed, so prose and tables cannot disagree.
    """
    best = extras["best_feature"]
    act = stats[stats.actionable]
    macros = {
        "nScreened":            ctx["n_screened"],
        "nEnrolled":            ctx["n_enrolled"],
        "nAnalysed":            ctx["n_analysed"],
        "nConfirmed":           ctx["n_confirmed"],
        "nUnconfirmed":         ctx["n_unconfirmed"],
        "dpnPrevalence":        f"{100 * ctx['prevalence']:.1f}\\%",
        "nPredictors":          ctx["n_modelled_features"],
        "nExcludedFeatures":    int((~stats.modelled).sum()),
        "nActionable":          int(stats.actionable.sum()),
        "eventsPerVariable":    f"{ctx['epv']:.1f}",
        "modelName":            ctx["model_name"],
        "modelAUC":             f"{ctx['model_auc']:.3f}",
        "bestFeature":          latex_escape(best.feature),
        "bestFeatureAUC":       f"{best.auc:.3f}",
        "nSignificantFeatures": extras["n_sig"],
        "actionableAUCmin":     f"{act.auc.min():.3f}",
        "actionableAUCmax":     f"{act.auc.max():.3f}",
        "ncsAUCmax":            f"{(stats[~stats.modelled].strength.max() + 0.5):.3f}",
        "nRecodedCells":        extras["n_recoded_cells"],
        "collinearPair":        f"{ctx['top_pair'][0]}--{ctx['top_pair'][1]}",
        "collinearRho":         f"{ctx['top_pair'][2]:.2f}",
    }
    lines = ["% Generated by edareport.py -- do not edit by hand.",
             f"% Source: {ctx['config_filename']}, produced {ctx['timestamp']}."]
    lines += [r"\newcommand{\%s}{%s}" % (name, value) for name, value in macros.items()]
    path = outputdir / "numbers.tex"
    path.write_text("\n".join(lines) + "\n")
    return path, macros


def collect_captions(outputdir):
    """Gather every standalone caption into one file for pasting into the manuscript."""
    order = ["table1_cohort_characteristics", "table2_feature_eligibility",
             "fig1a_participant_flow", "fig1bh_classification_composition",
             "fig1bv_classification_composition",
             "fig2_univariable_discrimination",
             "s1_continuous_rainclouds", "s2_categorical_features", "s3_redundancy",
             "s4_data_provenance", "s5_actionable_features", "s6_ncs_descriptives"]
    blocks = []
    for stem in order:
        path = outputdir / f"{stem}_caption.txt"
        if path.exists():
            blocks.append(f"% ---- {stem} " + "-" * (60 - len(stem)) + "\n" + path.read_text())
    combined = outputdir / "captions.txt"
    combined.write_text("\n".join(blocks))
    return combined


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python edareport.py <config file>")
        sys.exit(1)
    config_filename = sys.argv[1]

    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "experiments"
    config_dict = ymlconfig.load_config(config_path / config_filename)
    config = ymlconfig.dict_to_namespace(config_dict)

    outputdir = config_path / config.experiment.classification_type / config.reporting.output_subdir
    outputdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path / config_filename, outputdir / config_filename)

    rng = np.random.default_rng(config.experiment.random_seed)

    # ## Load and clean the data, keeping the loader's audit trail with the report
    D = DPN_data(str(script_dir / config.data.dataset_path))
    cleaning_report = outputdir / "cleaning_report.txt"
    D.load(classification=config.experiment.classification_type,
           report_path=str(cleaning_report))
    df = D.df
    target = D.get_target_column()

    inventory = build_inventory(D, config)
    stats = feature_statistics(df, inventory, target, config, rng)
    replacements, dropped = parse_cleaning_report(cleaning_report)

    # ## Reference performance of the fitted model, read from the optimization stage
    metrics = pd.read_csv(config_path / config.model_reference.metrics_path).set_index("metric")
    model_auc = float(metrics.loc["roc-auc", "mean"])

    # ## The most collinear pair among the modelling features, used by Figure S3
    modelled_features = stats[stats.modelled].feature.tolist()
    corr = df[modelled_features].corr(method="spearman")
    abs_corr = corr.abs().where(~np.eye(len(corr), dtype=bool))
    a, b = abs_corr.stack().idxmax()
    top_pair = (a, b, float(corr.loc[a, b]))

    n_analysed = len(df)
    n_confirmed = int((df[target] == 1).sum())
    n_unconfirmed = n_analysed - n_confirmed
    dropped_codes = [code for code, _ in dropped]
    ctx = {
        "n_screened":       config.cohort.screened,
        "n_enrolled":       config.cohort.enrolled,
        "n_analysed":       n_analysed,
        "n_confirmed":      n_confirmed,
        "n_unconfirmed":    n_unconfirmed,
        "prevalence":       n_confirmed / n_analysed,
        "dropped_codes":    dropped_codes,
        "dropped_codes_text": ", ".join(str(c) for c in dropped_codes),
        "screening_exclusion_reason": config.cohort.screening_exclusion_reason.strip(),
        "n_modelled_features": int(stats.modelled.sum()),
        "epv":              n_unconfirmed / int(stats.modelled.sum()),
        "model_auc":        model_auc,
        "model_name":       config.model_reference.name,
        "bootstrap_n":      config.analysis.bootstrap_n,
        "fisher_expected_min": config.analysis.fisher_expected_min,
        "vif_threshold":    config.analysis.vif_threshold,
        "figure_width_mm":  config.reporting.figure_width_mm,
        "top_pair":         top_pair,
        "config_filename":  config_filename,
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    formats = list(config.reporting.formats)

    print(f"  EDA | n={ctx['n_analysed']} | Confirmed={ctx['n_confirmed']} "
          f"| Unconfirmed={ctx['n_unconfirmed']} | predictors={ctx['n_modelled_features']}")
    print(f"  reference model: {ctx['model_name']} ROC-AUC {model_auc:.3f}")

    print("[Main text]")
    t1 = table1_cohort_characteristics(df, stats, target, outputdir, ctx)
    t2 = table2_feature_eligibility(stats, outputdir, ctx)
    f1a = figure1a_participant_flow(ctx, outputdir, formats)
    f1bh = figure1bh_classification_composition(ctx, outputdir, formats)
    f1bv = figure1bv_classification_composition(ctx, outputdir, formats)
    f2 = figure2_univariable_discrimination(stats, outputdir, ctx, formats)

    print("[Additional file 1]")
    s1 = s1_continuous_rainclouds(df, stats, target, outputdir, ctx, formats, rng)
    s2 = s2_categorical_features(df, stats, target, outputdir, ctx, formats)
    s3 = s3_redundancy(df, stats, outputdir, ctx, formats)
    s4 = s4_data_provenance(replacements, dropped, outputdir, ctx)
    s5 = s5_actionable_features(df, stats, target, outputdir, ctx, formats, rng)
    s6 = s6_ncs_descriptives(df, stats, target, outputdir, ctx)

    stats_csv = outputdir / "feature_statistics.csv"
    stats.to_csv(stats_csv, index=False)

    numbers_path, macros = write_numbers(outputdir, ctx, stats, {
        "best_feature": f2["best"], "n_sig": t1["n_sig"],
        "n_recoded_cells": s4["n_cells"]})
    captions_path = collect_captions(outputdir)

    manifest = {
        "generated": ctx["timestamp"],
        "config": config_filename,
        "cohort": {k: ctx[k] for k in ["n_screened", "n_enrolled", "n_analysed",
                                       "n_confirmed", "n_unconfirmed", "epv"]},
        "reference_model": {"name": ctx["model_name"], "roc_auc": model_auc},
        "macros": {k: str(v) for k, v in macros.items()},
        "artefacts": sorted(p.name for p in outputdir.iterdir() if p.is_file()),
    }
    (outputdir / "eda_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"  EDA complete. Outputs in: {outputdir}/")
    for label, result in [("Table 1", t1), ("Table 2", t2),
                          ("Figure 1a", f1a), ("Figure 1bh", f1bh), ("Figure 1bv", f1bv),
                          ("Figure 2", f2),
                          ("Figure S1", s1), ("Figure S2", s2), ("Figure S3", s3),
                          ("Table S1", s4), ("Figure S4", s5), ("Table S2", s6)]:
        names = result.get("paths") or [result.get("tex")]
        print(f"    {label:<10} {', '.join(Path(p).name for p in names)}")
    print(f"    macros     {numbers_path.name}")
    print(f"    captions   {captions_path.name}")
    print(f"    stats      {stats_csv.name}")


if __name__ == "__main__":
    main()
