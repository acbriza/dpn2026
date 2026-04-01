"""
Scientific Exploratory Data Analysis
=====================================
Produces publication-quality charts and tables for datasets with
numerical features and a binary categorical target/feature.

Requirements:
    pip install pandas numpy matplotlib seaborn scipy statsmodels

Usage:
    import pandas as pd
    from scientific_eda import run_eda

    df = pd.read_csv("your_data.csv")
    run_eda(df, target="your_binary_column", output_dir="eda_output")
"""
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


from pathlib import Path
import sys 
sys.path.append('..')  

from dataload import DPN_data
import ymlconfig


warnings.filterwarnings('ignore')


import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from scipy import stats
from scipy.stats import (
    shapiro, kstest, mannwhitneyu, chi2_contingency, pointbiserialr
)
from statsmodels.stats.multitest import multipletests
import itertools

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------

def set_publication_style():
    """Apply a clean, journal-ready matplotlib style."""
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif", "Georgia"],
        "font.size":          10,
        "axes.titlesize":     11,
        "axes.titleweight":   "bold",
        "axes.labelsize":     10,
        "axes.labelweight":   "bold",
        "axes.linewidth":     0.8,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.major.size":   4,
        "ytick.major.size":   4,
        "legend.fontsize":    9,
        "legend.frameon":     True,
        "legend.framealpha":  0.9,
        "legend.edgecolor":   "0.8",
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.facecolor":  "white",
        "lines.linewidth":    1.2,
        "patch.linewidth":    0.8,
    })

# Accessible two-class palette (colorblind-safe)
PALETTE = {"0": "#2166AC", "1": "#D6604D"}   # blue / red
PALETTE_LIST = ["#2166AC", "#D6604D"]
GREY = "#AAAAAA"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_columns(df, target, continuous):
    """Return lists of numerical and binary-categorical columns."""
    num_cols, cat_cols = [], []
    if continuous:
        num_cols = df.drop(target, axis=1).columns.to_list()
    else:
        cat_cols = df.drop(target, axis=1).columns.to_list()
    # for c in df.columns:
    #     if c == target:
    #         continue
    #     if pd.api.types.is_numeric_dtype(df[c]):
    #         num_cols.append(c)
    #     elif df[c].nunique() <= 2:
    #         cat_cols.append(c)
    #     # multi-class categoricals ignored (extend as needed)
    return num_cols, cat_cols


def _effect_size_r(u_stat, n1, n2):
    """Rank-biserial correlation r from Mann-Whitney U."""
    return 1 - (2 * u_stat) / (n1 * n2)


def _save(fig, path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path}")

# ---------------------------------------------------------------------------
# Table 1 — Descriptive statistics
# ---------------------------------------------------------------------------

def table_descriptive(df, num_cols, target, output_dir: Path):
    """
    Table 1: One row per numerical feature. Columns are grouped as:
      Range (Min, Max) | Mean ± SD (g0, g1, All) | Median [IQR] (g0, g1, All)
      | Stats (Skewness, Kurtosis, p (MW-U), r (effect size))
    """
    groups = df[target].unique()
    g0, g1 = sorted(groups)[:2]
    df0 = df[df[target] == g0]
    df1 = df[df[target] == g1]
 
    def _normality(v):
        if len(v) < 3:
            return float("nan")
        if len(v) <= 5000:
            _, p = shapiro(v)
        else:
            _, p = kstest(v, "norm", args=(v.mean(), v.std()))
        return p
 
    rows = {}
    for col in num_cols:
        v_all = df[col].dropna()
        v0    = df0[col].dropna()
        v1    = df1[col].dropna()
 
        # Mann-Whitney U
        if len(v0) > 0 and len(v1) > 0:
            u_stat, p_mw = mannwhitneyu(v0, v1, alternative="two-sided")
            r_eff = _effect_size_r(u_stat, len(v0), len(v1))
        else:
            p_mw, r_eff = float("nan"), float("nan")
 
        def _mean_sd(v):
            return f"{v.mean():.2f} ± {v.std():.2f}" if len(v) > 0 else "–"
 
        def _med_iqr(v):
            return (f"{v.median():.2f} [{v.quantile(0.25):.2f}–{v.quantile(0.75):.2f}]"
                    if len(v) > 0 else "–")
 
        rows[col] = {
            ("Range",        "Min"):              f"{v_all.min():.2f}" if len(v_all) > 0 else "–",
            ("Range",        "Max"):              f"{v_all.max():.2f}" if len(v_all) > 0 else "–",
            ("Mean ± SD",    str(g0)):            _mean_sd(v0),
            ("Mean ± SD",    str(g1)):            _mean_sd(v1),
            ("Mean ± SD",    "All"):              _mean_sd(v_all),
            ("Median [IQR]", str(g0)):            _med_iqr(v0),
            ("Median [IQR]", str(g1)):            _med_iqr(v1),
            ("Median [IQR]", "All"):              _med_iqr(v_all),
            ("Stats",        "Skewness"):         f"{stats.skew(v_all):.2f}"     if len(v_all) > 0 else "–",
            ("Stats",        "Kurtosis"):         f"{stats.kurtosis(v_all):.2f}" if len(v_all) > 0 else "–",
            ("Stats",        "p (MW-U)"):         f"{p_mw:.4f}"  if not np.isnan(p_mw)  else "–",
            ("Stats",        "r (effect size)"):  f"{r_eff:.3f}" if not np.isnan(r_eff) else "–",
        }
 
    if not rows:
        print("  Table 1 skipped — no numerical columns found.")
        return None
 
    tbl = pd.DataFrame.from_dict(rows, orient="index")
    tbl.index.name = None
    tbl.columns    = pd.MultiIndex.from_tuples(tbl.columns)
 
    csv_path = output_dir / "table1_descriptive.csv"
    tex_path = output_dir / "table1_descriptive.tex"
    tbl.to_csv(csv_path)
    tbl.to_latex(tex_path, longtable=True,
                 caption="Descriptive statistics of numerical features "
                         "stratified by binary outcome.",
                 label="tab:descriptive")
    print(f"  Table 1 saved → {csv_path}")
    print(f"  Table 1 LaTeX → {tex_path}")
    return tbl


# ---------------------------------------------------------------------------
# Table 2 — Categorical summary
# ---------------------------------------------------------------------------

def table_categorical(df, cat_cols, target, output_dir: Path):
    """
    Table 2: One row per categorical feature. Columns are grouped by category
    (g0, g1, Total, Percent) for each category value, followed by chi-square
    test statistics. The resulting DataFrame uses a MultiIndex column header.
    """
    if not cat_cols:
        return None
 
    groups = sorted(df[target].unique())
    g0, g1 = groups[0], groups[1]
    n_total_g0 = (df[target] == g0).sum()
    n_total_g1 = (df[target] == g1).sum()
    n_total    = len(df)
 
    # Collect all unique category values across all features to build columns
    all_cats = {}
    for col in cat_cols:
        all_cats[col] = sorted(df[col].dropna().unique())
 
    # Global category names across all features (union, for consistent columns)
    unique_cats = sorted({c for cats in all_cats.values() for c in cats})
 
    rows = {}
    for col in cat_cols:
        ct   = pd.crosstab(df[col], df[target])
        row  = {}
 
        for cat in unique_cats:
            if cat not in all_cats[col]:
                # Feature doesn't have this category — fill with dashes
                row[(str(cat), str(g0))]      = "–"
                row[(str(cat), str(g1))]      = "–"
                row[(str(cat), "Total")]      = "–"
                row[(str(cat), "Percent (%)")]= "–"
                continue
 
            n0  = int(ct.loc[cat, g0]) if g0 in ct.columns else 0
            n1  = int(ct.loc[cat, g1]) if g1 in ct.columns else 0
            tot = n0 + n1
            pct = f"{100 * tot / n_total:.1f}" if n_total > 0 else "–"
 
            row[(str(cat), str(g0))]       = n0
            row[(str(cat), str(g1))]       = n1
            row[(str(cat), "Total")]       = tot
            row[(str(cat), "Percent (%)")] = pct
 
        # Chi-square + Cramér's V
        chi2_val, p_chi, dof, _ = chi2_contingency(ct)
        n_obs    = ct.values.sum()
        v_cramer = np.sqrt(chi2_val / (n_obs * (min(ct.shape) - 1)))
 
        row[("Stats", "chi²")]      = f"{chi2_val:.2f}"
        row[("Stats", "df")]        = dof
        row[("Stats", "p (chi²)")]  = f"{p_chi:.4f}"
        row[("Stats", "Cramér's V")]= f"{v_cramer:.3f}"
 
        rows[col] = row
 
    tbl = pd.DataFrame.from_dict(rows, orient="index")
    tbl.index.name = None
    tbl.columns    = pd.MultiIndex.from_tuples(tbl.columns)
 
    csv_path = output_dir / "table2_categorical.csv"
    tex_path = output_dir / "table2_categorical.tex"
    tbl.to_csv(csv_path)
    tbl.to_latex(tex_path, longtable=True,
                 caption="Frequency distribution of categorical features "
                         "stratified by binary outcome.",
                 label="tab:categorical")
    print(f"  Table 2 saved → {csv_path}")
    return tbl

# ---------------------------------------------------------------------------
# Figure 1 — Distribution overview (histograms + KDE)
# ---------------------------------------------------------------------------

def fig_distributions(df, num_cols, target, output_dir: Path, filename_prefix: str,
                      custom_labels: dict | None = None,
                      grid_shape: tuple | None = None, 
                      title: str | None = None):
    """
    One panel per numerical feature: overlaid histograms + KDE curves for
    each class, with a rug plot and reported Mann-Whitney p-value.
    """
    set_publication_style()
    groups = sorted(df[target].unique())
    g0, g1 = groups[0], groups[1]
    if grid_shape is not None:
        n_rows_plot, n_cols_plot = grid_shape
    else:
        n_cols_plot = min(3, len(num_cols))
        n_rows_plot = int(np.ceil(len(num_cols) / n_cols_plot))

    fig, axes = plt.subplots(n_rows_plot, n_cols_plot,
                             figsize=(4.5 * n_cols_plot, 3.2 * n_rows_plot))
    axes = np.array(axes).flatten()

    for i, col in enumerate(num_cols):
        ax = axes[i]
        v0 = df[df[target] == g0][col].dropna()
        v1 = df[df[target] == g1][col].dropna()

        for v, g, c in [(v0, g0, PALETTE_LIST[0]), (v1, g1, PALETTE_LIST[1])]:
            ax.hist(v, bins=25, density=True, alpha=0.30, color=c, linewidth=0)
            kde_x = np.linspace(v.min(), v.max(), 300)
            kde   = stats.gaussian_kde(v)
            ax.plot(kde_x, kde(kde_x), color=c, lw=1.5, label=custom_labels['label_names']['target'][g])
            ax.plot(v, np.full(len(v), -0.002 * ax.get_ylim()[1]),
                    "|", color=c, alpha=0.3, markersize=3)

        # Mann-Whitney p
        if len(v0) > 0 and len(v1) > 0:
            _, p = mannwhitneyu(v0, v1, alternative="two-sided")
            p_str = f"p {'< 0.001' if p < 0.001 else f'= {p:.3f}'}"
            ax.text(0.97, 0.95, p_str, transform=ax.transAxes,
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="0.8", lw=0.6))

        ax.set_xlabel(custom_labels['target_name'])
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        # ax.set_title(col, pad=4)

    # Legend on last used axis
    handles = [plt.Line2D([0], [0], color=PALETTE_LIST[k], lw=2,
                           label=custom_labels['label_names']['target'][k]) for k in range(2)]
    axes[0].legend(handles=handles, loc="upper left", fontsize=8)

    # Hide unused panels
    for j in range(len(num_cols), len(axes)):
        axes[j].set_visible(False)

    if title: #"Distribution of Numerical Features by Binary Outcome"
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    path = output_dir / f"{filename_prefix}_distributions.png"
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# Figure 2 — Box-and-whisker with individual points (strip)
# ---------------------------------------------------------------------------

def fig_boxplots(df, num_cols, target, output_dir: Path, filename_prefix: str,
                 custom_labels: dict | None = None,
                 grid_shape: tuple | None = None, 
                 title: str | None = None):
    """
    Box-and-whisker plots with overlaid strip for each numerical feature,
    annotated with median and IQR.
    """
    set_publication_style()
    groups = sorted(df[target].unique())
    if grid_shape is not None:
        n_rows_plot, n_cols_plot = grid_shape
    else:
        n_cols_plot = min(3, len(num_cols))
        n_rows_plot = int(np.ceil(len(num_cols) / n_cols_plot))

    fig, axes = plt.subplots(n_rows_plot, n_cols_plot,
                             figsize=(4.0 * n_cols_plot, 3.5 * n_rows_plot))
    axes = np.array(axes).flatten()

    for i, col in enumerate(num_cols):
        ax = axes[i]
        data_plot = [df[df[target] == g][col].dropna().values for g in groups]
        labels    = [str(g) for g in groups]

        bp = ax.boxplot(data_plot, patch_artist=True, widths=0.45,
                        medianprops=dict(color="white", linewidth=1.8),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8),
                        flierprops=dict(marker="o", markersize=2,
                                        markerfacecolor=GREY, alpha=0.5,
                                        linewidth=0))
        for patch, color in zip(bp["boxes"], PALETTE_LIST):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        # Overlay strip
        for j, (vals, col_c) in enumerate(zip(data_plot, PALETTE_LIST)):
            jitter = np.random.normal(0, 0.06, size=len(vals))
            ax.scatter(np.full(len(vals), j + 1) + jitter, vals,
                       color=col_c, alpha=0.25, s=6, linewidths=0, zorder=2)

        # Annotate medians
        for j, vals in enumerate(data_plot):
            if len(vals) == 0:
                continue
            med = np.median(vals)
            ax.text(j + 1.27, med, f"{med:.1f}", va="center",
                    fontsize=7.5, color="0.3")

        ax.set_xticks([1, 2])
        ax.set_xticklabels(custom_labels['label_names']['target'])
        ax.set_xlabel(custom_labels['target_name'])
        ax.set_ylabel(col)
        # ax.set_title(col, pad=4)

    for j in range(len(num_cols), len(axes)):
        axes[j].set_visible(False)

    if title: # or "Box Plots of Numerical Features by Binary Outcome"
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    path = output_dir / f"{filename_prefix}_boxplots.png"
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# Figure 3 — Correlation heatmap
# ---------------------------------------------------------------------------

def fig_correlation_heatmap(df, num_cols, target, output_dir: Path):
    """
    Full Pearson/Spearman correlation matrix among numerical features,
    with lower-triangle mask and significance asterisks.
    """
    set_publication_style()
    sub = df[num_cols].copy()
    n   = len(num_cols)

    # Spearman (robust to non-normality)
    corr  = sub.corr(method="spearman")
    pvals = pd.DataFrame(np.ones((n, n)), index=num_cols, columns=num_cols)
    for a, b in itertools.combinations(num_cols, 2):
        pair = sub[[a, b]].dropna()          # drop rows where either is NaN
        _, p = stats.spearmanr(pair[a], pair[b])
        pvals.loc[a, b] = p
        pvals.loc[b, a] = p

    mask = np.triu(np.ones_like(corr, dtype=bool))  # hide upper triangle

    fig, ax = plt.subplots(figsize=(max(5, 0.7 * n), max(4, 0.6 * n)))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                linewidths=0.4, linecolor="white",
                annot=True, fmt=".2f", annot_kws={"size": 8},
                square=True, ax=ax,
                cbar_kws={"shrink": 0.7, "label": "Spearman ρ"})

    # Significance asterisks
    for i in range(n):
        for j in range(i):
            p = pvals.iloc[i, j]
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if stars:
                ax.text(j + 0.5, i + 0.78, stars, ha="center",
                        va="center", fontsize=7, color="white", fontweight="bold")

    ax.set_title("Figure 3. Spearman Correlation Matrix\n"
                 "(* p<0.05, ** p<0.01, *** p<0.001)",
                 fontweight="bold", pad=10)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    path = output_dir / "fig3_correlation_heatmap.png"
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# Figure 4 — Point-biserial correlation with target
# ---------------------------------------------------------------------------

def fig_point_biserial(df, num_cols, target, output_dir: Path):
    """
    Horizontal bar chart of point-biserial correlations between each
    numerical feature and the binary target, with 95% CI and BH-corrected p.
    """
    set_publication_style()
    y_bin = pd.to_numeric(df[target], errors="coerce").fillna(
        df[target].map({v: i for i, v in enumerate(sorted(df[target].unique()))}))

    results = []
    for col in num_cols:
        xy = df[[col, target]].dropna()
        y  = y_bin.loc[xy.index]
        r, p = pointbiserialr(y, xy[col])
        n    = len(xy)
        se   = np.sqrt((1 - r**2) / (n - 2))
        ci   = stats.t.ppf(0.975, df=n - 2) * se
        results.append({"feature": col, "r_pb": r, "ci": ci, "p": p, "n": n})

    res = pd.DataFrame(results).sort_values("r_pb")

    # BH correction
    _, p_adj, _, _ = multipletests(res["p"].values, method="fdr_bh")
    res["p_adj"] = p_adj

    colors = [PALETTE_LIST[1] if r > 0 else PALETTE_LIST[0]
              for r in res["r_pb"]]

    fig, ax = plt.subplots(figsize=(5.5, 0.5 * len(num_cols) + 1.5))
    bars = ax.barh(res["feature"], res["r_pb"], color=colors, alpha=0.75,
                   height=0.55)
    ax.errorbar(res["r_pb"], res["feature"], xerr=res["ci"],
                fmt="none", color="0.3", linewidth=1, capsize=3, capthick=0.8)
    ax.axvline(0, color="0.3", linewidth=0.8, linestyle="--")

    # Significance labels
    for _, row in res.iterrows():
        stars = ("***" if row["p_adj"] < 0.001 else
                 "**"  if row["p_adj"] < 0.01  else
                 "*"   if row["p_adj"] < 0.05  else "ns")
        x_pos = row["r_pb"] + row["ci"] + 0.01 * np.sign(row["r_pb"])
        ax.text(x_pos, row["feature"], stars, va="center", fontsize=8,
                ha="left" if row["r_pb"] >= 0 else "right")

    ax.set_xlabel("Point-Biserial Correlation (r$_{pb}$)")
    ax.set_title("Figure 4. Point-Biserial Correlations with Binary Outcome\n"
                 "(error bars = 95% CI; BH-adjusted: * p<0.05, ** p<0.01, *** p<0.001)",
                 fontweight="bold")
    ax.set_xlim(-1, 1)
    path = output_dir / "fig4_point_biserial.png"
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# Figure 5 — Missing value map
# ---------------------------------------------------------------------------

def fig_missing_values(df, num_cols, cat_cols, output_dir: Path):
    """
    Binary heatmap (present / missing) plus a bar chart of % missing
    per feature. Rows are a random sample of up to 200 observations.
    """
    set_publication_style()
    all_cols = num_cols + cat_cols
    sub = df[all_cols].copy()

    sample_n = min(200, len(sub))
    sub_s = sub.sample(sample_n, random_state=42)

    pct_missing = (sub.isnull().mean() * 100).sort_values(ascending=False)
    features_ordered = pct_missing.index.tolist()
    pct_missing = pct_missing[features_ordered]
    sub_s = sub_s[features_ordered]

    fig = plt.figure(figsize=(max(6, 0.5 * len(all_cols) + 2), 6))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.35)

    # Heatmap
    ax_heat = fig.add_subplot(gs[0])
    miss_mat = sub_s.isnull().astype(int)
    cmap_miss = plt.cm.colors.ListedColormap(["#F7F7F7", "#D6604D"])
    ax_heat.imshow(miss_mat.T, aspect="auto", cmap=cmap_miss,
                   interpolation="none", vmin=0, vmax=1)
    ax_heat.set_yticks(range(len(features_ordered)))
    ax_heat.set_yticklabels(features_ordered, fontsize=8)
    ax_heat.set_xlabel(f"Observations (random sample, n={sample_n})")
    ax_heat.set_title("Figure 5. Missing Value Map", fontweight="bold", pad=6)
    ax_heat.set_xticks([])
    # Legend
    from matplotlib.patches import Patch
    ax_heat.legend(handles=[Patch(fc="#F7F7F7", ec="0.5", label="Present"),
                             Patch(fc="#D6604D", ec="0.5", label="Missing")],
                   loc="upper right", fontsize=8, framealpha=0.9)

    # Bar chart
    ax_bar = fig.add_subplot(gs[1])
    bars = ax_bar.bar(features_ordered, pct_missing.values,
                      color="#D6604D", alpha=0.75, width=0.6)
    ax_bar.set_ylabel("% Missing")
    ax_bar.set_xticks(range(len(features_ordered)))
    ax_bar.set_xticklabels(features_ordered, rotation=45, ha="right", fontsize=8)
    ax_bar.set_ylim(0, max(pct_missing.values) * 1.25 + 1)
    for bar, pct in zip(bars, pct_missing.values):
        if pct > 0:
            ax_bar.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3, f"{pct:.1f}%",
                        ha="center", va="bottom", fontsize=7)

    path = output_dir / "fig5_missing_values.png"
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# Figure 6 — Q-Q plots (normality check)
# ---------------------------------------------------------------------------

def fig_qq_plots(df, num_cols, output_dir: Path):
    """
    Q-Q plots against the theoretical normal distribution for each
    numerical feature, with the Shapiro-Wilk p-value annotated.
    """
    set_publication_style()
    n_cols_plot = min(3, len(num_cols))
    n_rows_plot = int(np.ceil(len(num_cols) / n_cols_plot))

    fig, axes = plt.subplots(n_rows_plot, n_cols_plot,
                             figsize=(3.5 * n_cols_plot, 3.2 * n_rows_plot))
    axes = np.array(axes).flatten()

    for i, col in enumerate(num_cols):
        ax   = axes[i]
        vals = df[col].dropna()
        (osm, osr), (slope, intercept, _) = stats.probplot(vals, dist="norm")
        ax.scatter(osm, osr, color=PALETTE_LIST[0], s=10, alpha=0.55,
                   linewidths=0, label="Observed")
        x_line = np.array([osm[0], osm[-1]])
        ax.plot(x_line, slope * x_line + intercept, color=PALETTE_LIST[1],
                lw=1.2, label="Normal line")

        # SW test
        if len(vals) <= 5000:
            sw_p = shapiro(vals)[1]
            p_str = f"SW: p {'< 0.001' if sw_p < 0.001 else f'= {sw_p:.3f}'}"
        else:
            ks_p = kstest(vals, "norm", args=(vals.mean(), vals.std()))[1]
            p_str = f"KS: p {'< 0.001' if ks_p < 0.001 else f'= {ks_p:.3f}'}"

        ax.text(0.04, 0.93, p_str, transform=ax.transAxes, fontsize=8,
                va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                    ec="0.8", lw=0.6))
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Sample quantiles")
        ax.set_title(col, pad=4)
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(len(num_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Figure 6. Q-Q Plots for Normality Assessment",
                 fontsize=12, fontweight="bold", y=1.01)
    path = output_dir / "fig6_qq_plots.png"
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# Figure 7 — Pairplot (stratified scatter matrix)
# ---------------------------------------------------------------------------

def fig_pairplot(df, num_cols, target, output_dir: Path, max_features=6):
    """
    Stratified scatter matrix (upper: scatter, diagonal: KDE,
    lower: 2D KDE contours). Limited to `max_features` columns for legibility.
    """
    set_publication_style()
    cols_use = num_cols[:max_features]
    sub = df[cols_use + [target]].dropna()

    palette = {str(g): c for g, c in
               zip(sorted(df[target].unique()), PALETTE_LIST)}
    sub[target] = sub[target].astype(str)

    pg = sns.PairGrid(sub, vars=cols_use, hue=target,
                      palette=palette, corner=False,
                      diag_sharey=False)
    pg.map_upper(sns.scatterplot, s=8, alpha=0.35, linewidth=0)
    pg.map_diag(sns.kdeplot, fill=True, alpha=0.4, linewidth=1.0)
    pg.map_lower(sns.kdeplot, levels=5, alpha=0.7, linewidths=0.8)
    pg.add_legend(title=target, fontsize=9, title_fontsize=9)
    pg.figure.suptitle("Figure 7. Stratified Scatter Matrix",
                       fontsize=12, fontweight="bold", y=1.01)
    path = output_dir / "fig7_pairplot.png"
    pg.savefig(path)
    plt.close("all")
    print(f"  Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 8 — Categorical feature bar charts (stacked proportions)
# ---------------------------------------------------------------------------

def fig_categorical_bars(df, cat_cols, target, output_dir: Path):
    """
    For each binary categorical feature: a 100% stacked bar chart showing
    the proportion of each target class, with chi-square p annotated.
    """
    if not cat_cols:
        return None

    set_publication_style()
    groups = sorted(df[target].unique())
    n_cols_plot = min(3, len(cat_cols))
    n_rows_plot = int(np.ceil(len(cat_cols) / n_cols_plot))

    fig, axes = plt.subplots(n_rows_plot, n_cols_plot,
                             figsize=(4.0 * n_cols_plot, 3.2 * n_rows_plot))
    axes = np.array(axes).flatten()

    for i, col in enumerate(cat_cols):
        ax = axes[i]
        ct = pd.crosstab(df[col], df[target], normalize="index") * 100
        cats = ct.index.tolist()
        x    = np.arange(len(cats))
        bottom = np.zeros(len(cats))

        for g, color in zip(groups, PALETTE_LIST):
            vals = ct[g].values if g in ct.columns else np.zeros(len(cats))
            ax.bar(x, vals, bottom=bottom, color=color, alpha=0.75,
                   label=f"{target}={g}", width=0.5)
            # Label if segment > 8%
            for xi, (v, b) in enumerate(zip(vals, bottom)):
                if v > 8:
                    ax.text(xi, b + v / 2, f"{v:.0f}%",
                            ha="center", va="center", fontsize=8, color="white",
                            fontweight="bold")
            bottom += vals

        chi2, p_chi, _, _ = chi2_contingency(pd.crosstab(df[col], df[target]))
        p_str = f"χ² p {'< 0.001' if p_chi < 0.001 else f'= {p_chi:.3f}'}"
        ax.text(0.97, 0.97, p_str, transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", lw=0.6))

        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in cats])
        ax.set_ylabel("Proportion (%)")
        ax.set_xlabel(col)
        ax.set_title(col, pad=4)
        ax.set_ylim(0, 110)

    # Shared legend
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.75)
               for c in PALETTE_LIST]
    axes[0].legend(handles, [f"{target}={g}" for g in groups], fontsize=8)

    for j in range(len(cat_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Figure 8. Proportional Distribution of Categorical "
                 "Features by Outcome",
                 fontsize=12, fontweight="bold", y=1.01)
    path = output_dir / "fig8_categorical_bars.png"
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# Figure 9 — Outlier summary (Z-score heatmap)
# ---------------------------------------------------------------------------

def fig_outlier_heatmap(df, num_cols, output_dir: Path, z_thresh=3.0):
    """
    Heatmap where each cell = |z-score|. Cells exceeding z_thresh are
    highlighted, giving a quick per-observation outlier fingerprint.
    Rows are a random sample of up to 150 observations.
    """
    set_publication_style()
    sub = df[num_cols].dropna(how="all").copy()
    sub_s = sub.sample(min(150, len(sub)), random_state=42)
    z = np.abs(stats.zscore(sub_s, nan_policy="omit"))
    z_df = pd.DataFrame(z, columns=num_cols, index=sub_s.index)

    fig, ax = plt.subplots(figsize=(max(5, 0.65 * len(num_cols)),
                                    max(4, 0.06 * len(sub_s) + 2)))
    sns.heatmap(z_df.T, cmap="YlOrRd", vmin=0, vmax=z_thresh * 1.5,
                linewidths=0, ax=ax, cbar_kws={"label": "|Z-score|", "shrink": 0.7})
    ax.axhline(0, color="0.5", lw=0.4)
    ax.set_xlabel(f"Observations (n={len(sub_s)})")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_title(f"Figure 9. Outlier Map (|Z| > {z_thresh} flagged)\n"
                 "Darker cells indicate larger deviations from the mean",
                 fontweight="bold")
    path = output_dir / "fig9_outlier_heatmap.png"
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# Figure 10 — Violin plots
# ---------------------------------------------------------------------------

def fig_violins(df, num_cols, target, output_dir: Path):
    """
    Violin plots split by binary target, with inner box-and-whisker,
    median dot, and sample-size annotation.
    """
    set_publication_style()
    groups = sorted(df[target].unique())
    n_cols_plot = min(3, len(num_cols))
    n_rows_plot = int(np.ceil(len(num_cols) / n_cols_plot))

    fig, axes = plt.subplots(n_rows_plot, n_cols_plot,
                             figsize=(4.0 * n_cols_plot, 3.5 * n_rows_plot))
    axes = np.array(axes).flatten()

    for i, col in enumerate(num_cols):
        ax = axes[i]
        data_v = [df[df[target] == g][col].dropna().values for g in groups]

        parts = ax.violinplot(data_v, positions=[1, 2], widths=0.55,
                              showmedians=False, showextrema=False)
        for j, (pc, color) in enumerate(zip(parts["bodies"], PALETTE_LIST)):
            pc.set_facecolor(color)
            pc.set_alpha(0.55)
            pc.set_edgecolor("0.4")
            pc.set_linewidth(0.6)

        # Inner box
        for j, (vals, color) in enumerate(zip(data_v, PALETTE_LIST)):
            if len(vals) == 0:
                continue
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            ax.plot([j + 1, j + 1], [q1, q3], color="0.3", lw=2.5, solid_capstyle="round")
            ax.scatter([j + 1], [med], color="white", s=22, zorder=3,
                       edgecolors="0.3", linewidths=0.8)
            ax.text(j + 1, ax.get_ylim()[0] - 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                    f"n={len(vals)}", ha="center", va="top", fontsize=7.5, color="0.5")

        _, p = mannwhitneyu(data_v[0], data_v[1], alternative="two-sided") \
               if len(data_v[0]) > 0 and len(data_v[1]) > 0 else (None, None)
        if p is not None:
            p_str = f"p {'< 0.001' if p < 0.001 else f'= {p:.3f}'}"
            ax.text(0.97, 0.96, p_str, transform=ax.transAxes,
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", lw=0.6))

        ax.set_xticks([1, 2])
        ax.set_xticklabels([str(g) for g in groups])
        ax.set_xlabel(target)
        ax.set_ylabel(col)
        ax.set_title(col, pad=4)

    for j in range(len(num_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Figure 10. Violin Plots of Numerical Features by Outcome",
                 fontsize=12, fontweight="bold", y=1.01)
    path = output_dir / "fig10_violins.png"
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# Figure 11 — Bar plots (mean ± SE per class, with significance brackets)
# ---------------------------------------------------------------------------

def fig_bar(df, num_cols, target, output_dir: Path):
    """
    Bar plots of mean ± standard error for each numerical feature,
    grouped by the binary target. Significance brackets are drawn using
    Mann-Whitney U (BH-corrected), with stars above each pair.
    """
    set_publication_style()
    groups  = sorted(df[target].unique())
    g0, g1  = groups[0], groups[1]
    n_cols_plot = min(3, len(num_cols))
    n_rows_plot = int(np.ceil(len(num_cols) / n_cols_plot))

    # BH-correct p-values across all features up front
    pvals = []
    for col in num_cols:
        v0 = df[df[target] == g0][col].dropna()
        v1 = df[df[target] == g1][col].dropna()
        _, p = mannwhitneyu(v0, v1, alternative="two-sided") \
               if len(v0) > 0 and len(v1) > 0 else (None, 1.0)
        pvals.append(p)
    _, p_adj, _, _ = multipletests(pvals, method="fdr_bh")

    fig, axes = plt.subplots(n_rows_plot, n_cols_plot,
                             figsize=(4.0 * n_cols_plot, 3.5 * n_rows_plot))
    axes = np.array(axes).flatten()

    for i, col in enumerate(num_cols):
        ax = axes[i]
        v0 = df[df[target] == g0][col].dropna()
        v1 = df[df[target] == g1][col].dropna()

        means = [v0.mean(), v1.mean()]
        sems  = [v0.sem(),  v1.sem()]
        x     = np.array([0, 1])

        bars = ax.bar(x, means, yerr=sems, color=PALETTE_LIST, alpha=0.75,
                      width=0.5, capsize=4,
                      error_kw=dict(linewidth=0.9, capthick=0.9, ecolor="0.3"))

        # Significance bracket
        y_top   = max(m + s for m, s in zip(means, sems))
        y_bracket = y_top * 1.10
        y_tip     = y_top * 1.05
        ax.plot([0, 0, 1, 1],
                [y_tip, y_bracket, y_bracket, y_tip],
                color="0.3", linewidth=0.8)

        stars = ("***" if p_adj[i] < 0.001 else
                 "**"  if p_adj[i] < 0.01  else
                 "*"   if p_adj[i] < 0.05  else "ns")
        ax.text(0.5, y_bracket * 1.01, stars,
                ha="center", va="bottom", fontsize=9, color="0.2")

        # Annotate n per bar
        for xi, (g, v) in enumerate(zip(groups, [v0, v1])):
            ax.text(xi, -0.06 * y_bracket, f"n={len(v)}",
                    ha="center", va="top", fontsize=7.5, color="0.5")

        ax.set_xticks(x)
        ax.set_xticklabels([str(g) for g in groups])
        ax.set_xlabel(target)
        ax.set_ylabel(f"Mean {col}")
        ax.set_title(col, pad=4)
        ax.set_ylim(bottom=0,
                    top=y_bracket * 1.15 + max(sems) * 0.5)

    for j in range(len(num_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Figure 11. Mean ± SE of Numerical Features by Binary Outcome\n"
                 "(brackets: BH-adjusted MW-U; * p<0.05, ** p<0.01, *** p<0.001)",
                 fontsize=12, fontweight="bold", y=1.01)
    path = output_dir / "fig11_bar.png"
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# Figure 12 — Histograms (simple, no KDE overlay)
# ---------------------------------------------------------------------------

def fig_histogram(df, num_cols, target, output_dir: Path, filename_prefix: str,
                  custom_labels: dict | None = None,
                  grid_shape: tuple | None = None, 
                  title: str | None = None):
    """
    Plain histograms for each numerical feature, stratified by binary target.
    Bars for each class are plotted side-by-side. Bin count is set via the
    Freedman-Diaconis rule.
    """
    set_publication_style()
    groups = sorted(df[target].unique())
    g0, g1 = groups[0], groups[1]
    if grid_shape is not None:
        n_rows_plot, n_cols_plot = grid_shape
    else:
        n_cols_plot = min(3, len(num_cols))
        n_rows_plot = int(np.ceil(len(num_cols) / n_cols_plot))

    fig, axes = plt.subplots(n_rows_plot, n_cols_plot,
                             figsize=(4.5 * n_cols_plot, 3.2 * n_rows_plot))
    axes = np.array(axes).flatten()

    for i, col in enumerate(num_cols):
        ax = axes[i]
        v0 = df[df[target] == g0][col].dropna()
        v1 = df[df[target] == g1][col].dropna()

        # Freedman-Diaconis bin width on the full column
        v_all = df[col].dropna()
        iqr = v_all.quantile(0.75) - v_all.quantile(0.25)
        bin_width = 2 * iqr / (len(v_all) ** (1 / 3)) if iqr > 0 else 1
        n_bins = max(10, int(np.ceil((v_all.max() - v_all.min()) / bin_width)))
        bins = np.linspace(v_all.min(), v_all.max(), n_bins + 1)

        counts0, _ = np.histogram(v0, bins=bins)
        counts1, _ = np.histogram(v1, bins=bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bar_width   = (bins[1] - bins[0]) * 1.68

        ax.bar(bin_centers - bar_width / 2, counts0, width=bar_width,
               color=PALETTE_LIST[0], alpha=0.80, label=custom_labels['label_names']['target'][g0],
               linewidth=0.4, edgecolor="white")
        
        ax.bar(bin_centers + bar_width / 2, counts1, width=bar_width,
               color=PALETTE_LIST[1], alpha=0.80, label=custom_labels['label_names']['target'][g1],
               linewidth=0.4, edgecolor="white")

        ax.set_xticks([0,1])
        if col in custom_labels['label_names'].keys():
            ax.set_xticklabels(custom_labels['label_names'][col])
        else:
            ax.set_xticklabels(custom_labels['label_names']['default'])
        ax.set_ylabel("Count")
        ax.set_title(col, pad=4)
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(len(num_cols), len(axes)):
        axes[j].set_visible(False)

    if title: #"Histograms of Numerical Features by Binary Outcome"
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    path = output_dir / f"{filename_prefix}_histogram.png"
    _save(fig, path)
    return path




def run_eda(df: pd.DataFrame, 
            target: str, 
            output_dir: Path = Path("eda_output"),
            filename_prefix: str | None = None, 
            title: str | None = None, 
            custom_labels: dict | None = None, 
            grid_shape: tuple | None = None, 
            continuous: bool=True):
    """
    Run the full scientific EDA pipeline.

    Parameters
    ----------
    df         : Input DataFrame
    target     : Column name of the binary categorical outcome/feature
    output_dir : Path to directory where all outputs are saved (created if absent)
    title      : Optional custom suptitle applied to all figures
    grid_shape : Optional (n_rows, n_cols) tuple controlling the subplot grid
                 layout for multi-panel figures. Defaults to auto (max 3 cols).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Scientific EDA  |  n={len(df)}  |  target='{target}'")

    num_cols, cat_cols = _classify_columns(df, target, continuous, )
    print(f"Numerical features ({len(num_cols)}): {num_cols}")
    print(f"Binary categorical features ({len(cat_cols)}): {cat_cols}")
    print()

    print("[Tables]")
    table_descriptive(df, num_cols, target, output_dir)
    table_categorical(df, cat_cols, target, output_dir)

    print("\n[Figures]")
    if continuous:        
        fig_distributions(df, num_cols, target, output_dir, filename_prefix,
                        custom_labels=custom_labels, grid_shape=grid_shape, title=title)
        fig_boxplots(df, num_cols, target, output_dir, filename_prefix,
                    custom_labels=custom_labels, grid_shape=grid_shape, title=title)
    else:
        fig_histogram(df, cat_cols, target, output_dir, filename_prefix,
                    custom_labels=custom_labels, grid_shape=grid_shape, title=title)

    print(f"  EDA complete. All outputs in: {output_dir}/")


def demo():
    np.random.seed(0)
    N = 600
    outcome = np.random.choice([0, 1], size=N, p=[0.62, 0.38])

    demo_df = pd.DataFrame({
        "age":        np.where(outcome == 1,
                               np.random.normal(55, 12, N),
                               np.random.normal(45, 13, N)).clip(18, 90),
        "bmi":        np.where(outcome == 1,
                               np.random.normal(28.5, 5, N),
                               np.random.normal(25.0, 4, N)).clip(15, 55),
        "systolic_bp":np.where(outcome == 1,
                               np.random.normal(145, 20, N),
                               np.random.normal(125, 15, N)).clip(80, 220),
        "glucose":    np.where(outcome == 1,
                               np.random.gamma(shape=3, scale=40, size=N) + 80,
                               np.random.gamma(shape=2.5, scale=30, size=N) + 70),
        "cholesterol":np.random.normal(200, 40, N).clip(100, 350),
        "creatinine": np.abs(np.random.normal(1.1, 0.4, N)),
        "sex":        np.random.choice([0, 1], N),
        "smoker":     np.where(outcome == 1,
                               np.random.choice([0, 1], N, p=[0.45, 0.55]),
                               np.random.choice([0, 1], N, p=[0.72, 0.28])),
        "outcome":    outcome,
    })

    # Introduce ~4 % missing values
    for col in ["bmi", "glucose", "creatinine"]:
        idx = np.random.choice(demo_df.index, size=int(0.04 * N), replace=False)
        demo_df.loc[idx, col] = np.nan

    run_eda(demo_df, target="outcome", output_dir=Path("eda_output_demo"))

# ---------------------------------------------------------------------------
# Demo — runs with synthetic data when executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ## Read Config File    
    current_file = Path(__file__).resolve() # Get the absolute path of the current file
    script_dir = current_file.parent # Get the directory containing the file

    config_path = Path(script_dir /'experiments')    
    config_filename =  "bin_cf_final.yml"
    config_dict = ymlconfig.load_config(config_path / config_filename)
    config = ymlconfig.dict_to_namespace(config_dict)
    print(config_dict)

    outputdir = config_path /  config.experiment.classification_type /  'eda'
    outputdir.mkdir(parents=True, exist_ok=True)
    print(outputdir)    

    D = DPN_data(config.data.dataset_path[3:])
    D.load(classification=config.experiment.classification_type)

    dfdpn = D.df
    data_cols = dfdpn.drop(D.non_data_cols, axis=1, errors="ignore").columns
    X = dfdpn[data_cols]
    target_col = "Confirmed_Binary_DPN"
    y = dfdpn[target_col]
    print(X.shape, y.shape)    

    dfXy = pd.concat([X, y], axis=1)
    print(X.shape, y.shape, dfXy.shape)


    custom_labels = {
        "target_name" : "DPN Type",
        "label_names" : {
            "target" : ["Unconfirmed", "Confirmed"],
            "SEX" : ["Male", "Female"],
            "default" : ["Yes", "No"],
        }
    }

    cols = ['SEX', 'SUBJ', 'INSULIN'] + D.comorbidity_cols + D.neuro_cols + [target_col]
    run_eda(dfXy[cols], 
                target=target_col,
                filename_prefix='categorical', 
                output_dir=outputdir/'categorical',
                # title="Binary Data from Profile, Commorbidity, Neurological Study",
                custom_labels=custom_labels,
                grid_shape=(4,3),
                continuous=False,
                )

    cols = ['AGE', 'DM_DUR', 'HBA1C', 'MNSI'] + [target_col]
    run_eda(dfXy[cols], 
                target=target_col, 
                filename_prefix='profile', 
                output_dir=outputdir/'profile',
                # title="Continuous Data from Profile and MNSI Data",
                custom_labels=custom_labels,
                grid_shape=(1,4),
                continuous=True,
                )

    cols = D.ncs_cols + [target_col]
    run_eda(dfXy[cols], 
                target=target_col, 
                filename_prefix='ncs', 
                output_dir=outputdir/'ncs',
                # title="Nerve Conduction Studies",
                custom_labels=custom_labels,
                grid_shape=(6,3),
                continuous=True
                )

    cols = D.sudo_cols + [target_col]
    run_eda(dfXy[cols], 
                target=target_col, 
                output_dir=outputdir/'sudo',
                filename_prefix='sudo', 
                # title="Sudoscan",
                custom_labels=custom_labels,
                grid_shape=(2,3),
                continuous=True
                )