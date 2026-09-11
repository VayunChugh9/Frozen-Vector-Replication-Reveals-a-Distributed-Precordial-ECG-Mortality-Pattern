"""Generate the submission figure set from saved machine-readable results.

The graphics deliberately contain only reading aids (axes, ticks, legends, and
panel labels). Statistical interpretation and exact numerical results live in
the manuscript captions and saved tables.
"""
from __future__ import annotations

from pathlib import Path
import json

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FAMILIES = ["rs_diff", "rs_diff_2"]
FEATURES = [f"{family}_{lead}" for lead in LEADS for family in FAMILIES]

# One restrained, color-blind-friendly visual system for the entire paper.
BLUE = "#3267A8"
CORAL = "#D5675D"
TEAL = "#2A8C82"
GOLD = "#C9952C"
INK = "#25313C"
GRAY = "#7C8793"
LIGHT = "#DDE3E9"

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.3,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.7,
        "ytick.labelsize": 6.7,
        "legend.fontsize": 6.7,
        "axes.linewidth": 0.55,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.15,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 180,
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def clean(ax, *, top=False, right=False):
    ax.spines["top"].set_visible(top)
    ax.spines["right"].set_visible(right)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)


def panel(ax, label, x=-0.14, y=1.05):
    ax.text(x, y, f"({label})", transform=ax.transAxes, fontsize=8.2,
            fontweight="bold", ha="left", va="bottom", color=INK)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.025, dpi=400)
    plt.close(fig)


def corrected_coefficients():
    coeff = pd.read_csv(R / "adjustment_robustness_coefficients.csv")
    coeff = coeff.query("specification == 'full'")
    discovery = coeff.query("cohort == 'CODE_discovery'").set_index("feature").loc[FEATURES]
    replication = coeff.query("cohort == 'CODE_replication_corrected'").set_index("feature").loc[FEATURES]
    return discovery, replication


def fig1_topology():
    discovery, replication = corrected_coefficients()
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.30), sharey=True)
    x = np.arange(len(LEADS))
    for k, (ax, family) in enumerate(zip(axes, FAMILIES)):
        feats = [f"{family}_{lead}" for lead in LEADS]
        ax.axhline(0, color=LIGHT, linewidth=0.7, zorder=0)
        ax.plot(x, discovery.loc[feats, "log_hr"], color=BLUE, marker="o", markersize=3.2,
                label="Discovery", zorder=3)
        ax.plot(x, replication.loc[feats, "log_hr"], color=CORAL, marker="s", markersize=3.0,
                linestyle="--", label="Replication", zorder=3)
        ax.set_xticks(x, LEADS, rotation=45, ha="right")
        ax.set_xlabel("ECG lead")
        ax.set_ylim(-0.17, 0.10)
        clean(ax)
        panel(ax, chr(ord("a") + k), x=-0.13)
    axes[0].set_ylabel("Log HR per SD")
    axes[0].legend(frameon=False, ncol=2, loc="upper left", handlelength=1.7,
                   columnspacing=1.0, borderaxespad=0.2)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.25, top=0.96, wspace=0.18)
    save(fig, "fig1_topology")


def fig2_replication():
    discovery, replication = corrected_coefficients()
    fig, ax = plt.subplots(figsize=(3.35, 3.18))
    for family, color, marker, label in [
        ("rs_diff", BLUE, "o", "First difference"),
        ("rs_diff_2", CORAL, "s", "Second difference"),
    ]:
        feats = [f"{family}_{lead}" for lead in LEADS]
        ax.scatter(discovery.loc[feats, "log_hr"], replication.loc[feats, "log_hr"],
                   s=22, color=color, marker=marker, edgecolor="white", linewidth=0.35,
                   label=label, zorder=3)
    lo = min(discovery.log_hr.min(), replication.log_hr.min()) - 0.012
    hi = max(discovery.log_hr.max(), replication.log_hr.max()) + 0.012
    ax.plot([lo, hi], [lo, hi], color=GRAY, linewidth=0.75, linestyle=(0, (2, 2)), zorder=1)
    ax.axhline(0, color=LIGHT, linewidth=0.55, zorder=0)
    ax.axvline(0, color=LIGHT, linewidth=0.55, zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Discovery log HR per SD")
    ax.set_ylabel("Replication log HR per SD")
    ax.legend(frameon=False, loc="lower right", handletextpad=0.45, borderaxespad=0.2)
    clean(ax, top=True, right=True)
    fig.subplots_adjust(left=0.20, right=0.985, bottom=0.16, top=0.985)
    save(fig, "fig2_replication")


def fig3_transfer():
    transfer = pd.read_csv(R / "frozen_score_transfer.csv")
    order = ["full_12lead", "precordial_only", "limb_only", "full_minus_aVL", "full_minus_V1_V3"]
    labels = ["Full 12-lead", "Precordial only", "Limb only", "Without aVL", "Without V1–V3"]
    frame = (transfer.query("cohort == 'CODE_replication_corrected'")
             .drop_duplicates("score").set_index("score").loc[order])
    y = np.arange(len(frame))[::-1]
    colors = [BLUE, TEAL, GRAY, GOLD, CORAL]
    fig, ax = plt.subplots(figsize=(3.35, 2.30))
    for yy, (_, row), color in zip(y, frame.iterrows(), colors):
        ax.errorbar(row.hr_per_sd, yy,
                    xerr=[[row.hr_per_sd - row.ci95_lower], [row.ci95_upper - row.hr_per_sd]],
                    fmt="o", markersize=4.2, color=color, ecolor=color,
                    elinewidth=1.0, capsize=2.0, capthick=0.7, zorder=3)
    ax.axvline(1, color=GRAY, linewidth=0.7, linestyle=(0, (2, 2)), zorder=0)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Hazard ratio per score SD")
    ax.set_xlim(0.90, 1.26)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
    ax.tick_params(axis="y", length=0)
    clean(ax)
    fig.subplots_adjust(left=0.36, right=0.98, bottom=0.20, top=0.98)
    save(fig, "fig3_frozen_transfer")


def fig4_null_bootstrap():
    draws = pd.read_csv(R / "topology_outcome_permutation_draws.csv")
    observed = pd.read_csv(R / "topology_replication_metrics_corrected.csv").iloc[0]
    boot = pd.read_csv(R / "replication_bootstrap_draws.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.32), gridspec_kw={"width_ratios": [1, 1, 1.18]})
    for i, (ax, column, xlabel) in enumerate([
        (axes[0], "cosine_similarity", "Cosine similarity"),
        (axes[1], "projection", "Discovery-axis projection"),
    ]):
        ax.hist(draws[column], bins=32, color=LIGHT, edgecolor="white", linewidth=0.35,
                label="Permuted")
        ax.axvline(observed[column], color=CORAL, linewidth=1.45, label="Observed")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Permutation count" if i == 0 else "")
        clean(ax)
        panel(ax, chr(ord("a") + i), x=-0.17)
    axes[0].legend(frameon=False, loc="upper left", borderaxespad=0.15, handlelength=1.4)

    columns = ["pearson_r", "spearman_rho", "cosine_similarity", "projection", "sign_concordance"]
    names = ["Pearson", "Spearman", "Cosine", "Projection", "Sign fraction"]
    quantiles = boot[columns].quantile([0.025, 0.5, 0.975])
    y = np.arange(len(columns))[::-1]
    ax = axes[2]
    for yy, column in zip(y, columns):
        median = quantiles.loc[0.5, column]
        ax.errorbar(median, yy,
                    xerr=[[median - quantiles.loc[0.025, column]],
                          [quantiles.loc[0.975, column] - median]],
                    fmt="o", color=BLUE, markersize=3.8, elinewidth=0.95,
                    capsize=1.8, capthick=0.7)
    ax.axvline(0, color=LIGHT, linewidth=0.55)
    ax.set_yticks(y, names)
    ax.set_xlabel("Bootstrap estimate")
    ax.set_xlim(0, 1.0)
    ax.tick_params(axis="y", length=0)
    clean(ax)
    panel(ax, "c", x=-0.25)
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.22, top=0.96, wspace=0.42)
    save(fig, "fig4_null_bootstrap")


def fig5_robustness():
    robustness = pd.read_csv(R / "adjustment_robustness_metrics.csv").set_index("specification")
    sensitivity = pd.read_csv(R / "topology_sensitivity_metrics_primary.csv").set_index("analysis")
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.40))
    metrics = ["pearson_r", "spearman_rho", "cosine_similarity", "sign_concordance"]
    labels = ["Pearson", "Spearman", "Cosine", "Signs"]
    x = np.arange(len(metrics))
    for key, label, color, marker in [
        ("unadjusted", "Unadjusted", GRAY, "o"),
        ("age_sex", "Age + sex", TEAL, "s"),
        ("full", "Full", CORAL, "^"),
    ]:
        axes[0].plot(x, robustness.loc[key, metrics], marker=marker, markersize=3.8,
                     color=color, label=label)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.25, 1.02)
    axes[0].set_ylabel("Replication metric")
    axes[0].legend(frameon=False, ncol=3, loc="lower left", handlelength=1.5,
                   columnspacing=0.8, borderaxespad=0.15)
    clean(axes[0])
    panel(axes[0], "a")

    order = ["original_zero_time_inclusive_audit", "adjusted_for_aVL_QS_span",
             "feature_trim_0.5pct_each_tail", "corrected_primary"]
    labels = ["Original audit", "Q–S-span adjusted", "0.5% tail trim", "Corrected primary"]
    colors = [GRAY, GOLD, TEAL, BLUE]
    y = np.arange(len(order))[::-1]
    for yy, key, color in zip(y, order, colors):
        axes[1].scatter(sensitivity.loc[key, "cosine_similarity"], yy, s=25,
                        color=color, edgecolor="white", linewidth=0.35, zorder=3)
    axes[1].set_yticks(y, labels)
    axes[1].set_xlim(0.62, 0.88)
    axes[1].set_xlabel("Cosine similarity")
    axes[1].tick_params(axis="y", length=0)
    clean(axes[1])
    panel(axes[1], "b", x=-0.28)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.20, top=0.96, wspace=0.48)
    save(fig, "fig5_robustness")


def figS1_subgroups():
    table = pd.read_csv(R / "final_results_table.csv")
    subset = table.query("feature in ['rs_diff_V1','rs_diff_V2','rs_diff_V3'] and subgroup != 'all'").copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.25), sharex=True)
    for ax, subgroup in zip(axes, ["normal_ecg", "rhythm_abnormalities_excluded"]):
        y = np.arange(3)[::-1]
        for cohort, label, color, marker, shift in [
            ("CODE_discovery", "Discovery", BLUE, "o", 0.09),
            ("CODE_replication_corrected", "Replication", CORAL, "s", -0.09),
        ]:
            frame = (subset.query("cohort == @cohort and subgroup == @subgroup")
                     .set_index("feature").loc[["rs_diff_V1", "rs_diff_V2", "rs_diff_V3"]])
            ax.errorbar(frame.hr_per_sd, y + shift,
                        xerr=[frame.hr_per_sd - frame.ci95_lower, frame.ci95_upper - frame.hr_per_sd],
                        fmt=marker, markersize=3.7, color=color, ecolor=color, elinewidth=0.9,
                        capsize=1.7, label=label)
        ax.axvline(1, color=GRAY, linewidth=0.65, linestyle=(0, (2, 2)))
        ax.set_yticks(y, ["V1", "V2", "V3"])
        ax.set_xlabel("Hazard ratio per SD")
        clean(ax)
    axes[0].set_ylabel("ECG lead")
    axes[0].legend(frameon=False, loc="lower left", ncol=2,
                   bbox_to_anchor=(0.0, 1.01), borderaxespad=0.0)
    panel(axes[0], "a")
    panel(axes[1], "b")
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.22, top=0.96, wspace=0.18)
    save(fig, "figS1_subgroups")


def figS2_sami():
    table = pd.read_csv(R / "final_results_table.csv")
    discovery = table.query("cohort == 'CODE_discovery' and subgroup == 'all'").set_index("feature").loc[FEATURES]
    sami = table.query("cohort == 'SaMi_Trop_supportive' and subgroup == 'all'").set_index("feature").loc[FEATURES]
    x = np.log(discovery.hr_per_sd)
    y = np.log(sami.hr_per_sd)
    concordant = np.sign(x) == np.sign(y)
    fig, ax = plt.subplots(figsize=(3.35, 3.12))
    ax.scatter(x[concordant], y[concordant], s=21, color=BLUE, marker="o",
               edgecolor="white", linewidth=0.35, label="Concordant")
    ax.scatter(x[~concordant], y[~concordant], s=25, color=CORAL, marker="x",
               linewidth=1.0, label="Opposite sign")
    ax.axhline(0, color=LIGHT, linewidth=0.55)
    ax.axvline(0, color=LIGHT, linewidth=0.55)
    ax.set_xlabel("CODE discovery log HR per SD")
    ax.set_ylabel("SaMi-Trop log HR per SD")
    ax.legend(frameon=False, loc="lower right", borderaxespad=0.2)
    clean(ax, top=True, right=True)
    fig.subplots_adjust(left=0.20, right=0.985, bottom=0.16, top=0.985)
    save(fig, "figS2_sami")


def figS3_validity():
    sensitivity = pd.read_csv(R / "artifact_qrs_duration_sensitivity_primary.csv")
    sensitivity = sensitivity.query(
        "cohort == 'CODE_replication_corrected' and feature in ['rs_diff_aVL','rs_diff_V1','rs_diff_V2','rs_diff_V3']"
    )
    fig, ax = plt.subplots(figsize=(3.35, 2.38))
    y = np.arange(4)[::-1]
    features = ["rs_diff_aVL", "rs_diff_V1", "rs_diff_V2", "rs_diff_V3"]
    for analysis, label, color, marker, shift in [
        ("adjusted_for_aVL_QS_span", "Q–S-span adjusted", GOLD, "o", 0.09),
        ("feature_trim_0.5pct_each_tail", "0.5% tail trim", TEAL, "s", -0.09),
    ]:
        frame = sensitivity.query("analysis == @analysis").set_index("feature").loc[features]
        ax.errorbar(frame.hr_per_sd, y + shift,
                    xerr=[frame.hr_per_sd - frame.ci95_lower, frame.ci95_upper - frame.hr_per_sd],
                    fmt=marker, markersize=3.7, color=color, ecolor=color,
                    elinewidth=0.9, capsize=1.7, label=label)
    ax.axvline(1, color=GRAY, linewidth=0.65, linestyle=(0, (2, 2)))
    ax.set_yticks(y, ["aVL", "V1", "V2", "V3"])
    ax.set_xlabel("Hazard ratio per SD")
    ax.legend(frameon=False, loc="lower left", ncol=2,
              bbox_to_anchor=(0.0, 1.01), borderaxespad=0.0,
              handlelength=1.5, columnspacing=0.8)
    ax.tick_params(axis="y", length=0)
    clean(ax)
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.19, top=0.98)
    save(fig, "figS3_validity")


def figS4_anatomy():
    anatomy = pd.read_csv(R / "anatomy_topology_results.csv").set_index("feature_family").loc[FAMILIES]
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.28))
    x = np.arange(2)
    for label, column, color, marker, shift in [
        ("Discovery", "discovery_energy", BLUE, "o", -0.07),
        ("Replication", "replication_energy", CORAL, "s", 0.07),
    ]:
        axes[0].scatter(x + shift, anatomy[column], s=27, color=color, marker=marker,
                        edgecolor="white", linewidth=0.35, label=label, zorder=3)
    axes[0].set_xticks(x, ["First difference", "Second difference"])
    axes[0].set_ylabel("Normalized graph energy")
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(frameon=False, loc="upper left", borderaxespad=0.15)
    clean(axes[0])
    panel(axes[0], "a")
    axes[1].scatter(x, anatomy.alignment_cosine, s=29, color=[TEAL, GOLD],
                    edgecolor="white", linewidth=0.35, zorder=3)
    axes[1].set_xticks(x, ["First difference", "Second difference"])
    axes[1].set_ylabel("Anatomy-preserving alignment")
    axes[1].set_ylim(0, 1.0)
    clean(axes[1])
    panel(axes[1], "b")
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.20, top=0.96, wspace=0.27)
    save(fig, "figS4_anatomy")


def figS5_schoenfeld():
    ph = pd.read_csv(R / "schoenfeld_ph_tests.csv")
    fig, ax = plt.subplots(figsize=(7.05, 2.20))
    x = np.arange(len(FEATURES))
    for cohort, label, color, marker, shift in [
        ("CODE_discovery", "Discovery", BLUE, "o", -0.10),
        ("CODE_replication_corrected", "Replication", CORAL, "s", 0.10),
    ]:
        frame = ph.query("cohort == @cohort").set_index("feature").loc[FEATURES]
        ax.scatter(x + shift, -np.log10(frame.feature_schoenfeld_p), s=15,
                   color=color, marker=marker, label=label, zorder=3)
    ax.axhline(-np.log10(0.05 / 24), color=GRAY, linewidth=0.7, linestyle=(0, (3, 2)),
               label="0.05/24")
    tick_labels = [f"{lead}\n{'1st' if family == 'rs_diff' else '2nd'}" for lead in LEADS for family in FAMILIES]
    ax.set_xticks(x, tick_labels, rotation=90)
    ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.legend(frameon=False, ncol=3, loc="upper left", borderaxespad=0.1)
    clean(ax)
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.35, top=0.97)
    save(fig, "figS5_schoenfeld")


def _median_beat_scipy(signal, fs=400):
    """Median beat for visualization only, using deterministic local peak detection."""
    signal = np.asarray(signal, dtype=float)
    ref = signal[:, 4]
    centered = ref - np.median(ref)
    peaks, _ = find_peaks(np.abs(centered), distance=int(0.40 * fs),
                          prominence=max(np.std(centered) * 0.7, 1e-8))
    half = int(0.50 * fs)
    peaks = peaks[(peaks >= half) & (peaks + half <= len(ref))]
    if len(peaks) == 0:
        return None
    beats = np.stack([signal[p - half:p + half] for p in peaks])
    return np.median(beats, axis=0)


def build_sami_waveform_summary():
    output = R / "sami_frozen_score_waveform_summary.csv"
    manifest_path = R / "sami_frozen_score_waveform_manifest.json"
    features = pd.read_parquet(R / "sami_features.parquet").query("not qc_fail_any").copy()
    parameters = pd.read_csv(R / "frozen_score_parameters.csv").set_index("feature")
    score = np.zeros(len(features))
    for feature, row in parameters.iterrows():
        score += row.discovery_beta * (features[feature].to_numpy() - row.discovery_mean) / row.discovery_sd
    features["frozen_score"] = score
    low_cut, high_cut = features.frozen_score.quantile([0.20, 0.80])
    features["score_group"] = np.where(features.frozen_score <= low_cut, "Lower quintile",
                                        np.where(features.frozen_score >= high_cut, "Upper quintile", ""))
    selected = features.query("score_group != ''")
    group_beats = {"Lower quintile": [], "Upper quintile": []}
    source_h5 = ROOT / "data" / "sami" / "exams.hdf5"
    with h5py.File(source_h5, "r") as handle:
        tracings = handle["tracings"]
        for row in selected.itertuples():
            beat = _median_beat_scipy(tracings[int(row.hdf5_row_idx)])
            if beat is None or beat.shape != (400, 12):
                continue
            lo = np.nanmin(beat, axis=0)
            span = np.nanmax(beat, axis=0) - lo
            if np.any(~np.isfinite(span)) or np.any(span <= 1e-12):
                continue
            group_beats[row.score_group].append((beat - lo) / span)
    records = []
    for group, beats in group_beats.items():
        stack = np.stack(beats)
        for lead_index, lead in enumerate(LEADS):
            curves = stack[:, :, lead_index]
            for sample in range(curves.shape[1]):
                records.append({
                    "lead": lead,
                    "time_ms": (sample - 200) / 400 * 1000,
                    "score_group": group,
                    "median_normalized_amplitude": np.median(curves[:, sample]),
                    "q25_normalized_amplitude": np.quantile(curves[:, sample], 0.25),
                    "q75_normalized_amplitude": np.quantile(curves[:, sample], 0.75),
                    "n_waveforms": len(beats),
                })
    pd.DataFrame(records).to_csv(output, index=False)
    manifest = {
        "source": str(source_h5.relative_to(ROOT)),
        "selection": "SaMi-Trop extraction-QC pass; lower and upper quintiles of frozen CODE discovery score",
        "outcome_used_for_selection": False,
        "alignment": "aVL absolute-peak aligned median beats; 400 Hz; -500 to +497.5 ms",
        "scaling": "per-record, per-lead min-max normalization matching feature-shape scale",
        "n_lower": len(group_beats["Lower quintile"]),
        "n_upper": len(group_beats["Upper quintile"]),
        "machine_readable_summary": str(output.relative_to(ROOT)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return pd.read_csv(output)


def figS6_waveforms():
    data = build_sami_waveform_summary()
    fig, axes = plt.subplots(3, 4, figsize=(7.05, 4.65), sharex=True, sharey=True)
    for ax, lead in zip(axes.flat, LEADS):
        for group, color, linestyle in [
            ("Lower quintile", BLUE, "-"),
            ("Upper quintile", CORAL, "--"),
        ]:
            frame = data.query("lead == @lead and score_group == @group").sort_values("time_ms")
            ax.plot(frame.time_ms, frame.median_normalized_amplitude, color=color,
                    linestyle=linestyle, linewidth=1.05, label=group)
            ax.fill_between(frame.time_ms, frame.q25_normalized_amplitude,
                            frame.q75_normalized_amplitude, color=color, alpha=0.08, linewidth=0)
        ax.text(0.04, 0.91, lead, transform=ax.transAxes, fontsize=7.0, color=INK,
                ha="left", va="top")
        ax.set_xlim(-250, 400)
        ax.set_ylim(0.05, 0.95)
        ax.set_xticks([-200, 0, 200, 400])
        clean(ax)
    axes[0, 0].legend(frameon=False, ncol=2, loc="lower left",
                      bbox_to_anchor=(0.0, 1.04), borderaxespad=0, handlelength=1.7)
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Normalized amplitude")
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.09, top=0.92, wspace=0.16, hspace=0.20)
    save(fig, "figS6_waveforms")


def fig6_waveforms_focal():
    """Single-column aVL/V1--V3 waveform panel for the main paper."""
    data = pd.read_csv(R / "sami_frozen_score_waveform_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(3.35, 2.62), sharex=True, sharey=True)
    for ax, lead in zip(axes.flat, ["aVL", "V1", "V2", "V3"]):
        for group, color, linestyle in [
            ("Lower quintile", BLUE, "-"),
            ("Upper quintile", CORAL, "--"),
        ]:
            frame = data.query("lead == @lead and score_group == @group").sort_values("time_ms")
            ax.plot(frame.time_ms, frame.median_normalized_amplitude, color=color,
                    linestyle=linestyle, linewidth=1.0, label=group)
            ax.fill_between(frame.time_ms, frame.q25_normalized_amplitude,
                            frame.q75_normalized_amplitude, color=color, alpha=0.08, linewidth=0)
        ax.text(0.05, 0.90, lead, transform=ax.transAxes, fontsize=6.8, color=INK,
                ha="left", va="top")
        ax.set_xlim(-200, 350)
        ax.set_ylim(0.05, 0.95)
        ax.set_xticks([-200, 0, 200])
        clean(ax)
    axes[0, 0].legend(frameon=False, ncol=2, loc="lower left",
                      bbox_to_anchor=(0.0, 1.03), borderaxespad=0,
                      handlelength=1.5, columnspacing=0.8)
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Normalized amplitude")
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.13, top=0.90, wspace=0.15, hspace=0.18)
    save(fig, "fig6_waveforms")


CAPTIONS = {
    "fig1_topology": "Fully adjusted discovery and positive-follow-up replication log hazard ratios across the 12 leads for (a) first-difference and (b) second-difference terminal-QRS features. The complete 24-coefficient vector is the prespecified replication target; points are not filtered by statistical significance.",
    "fig2_replication": "Discovery versus positive-follow-up replication coefficients for all 24 fixed terminal-QRS features. The diagonal denotes equality. Whole-vector agreement was Pearson r=0.550, Spearman rho=0.354, cosine similarity=0.776, with 20/24 concordant signs.",
    "fig3_frozen_transfer": "Frozen-score transfer and prespecified ablations in the corrected replication cohort. Points are hazard ratios per evaluation-cohort SD and horizontal bars are model-based 95% confidence intervals. The full and precordial-only scores transferred, whereas limb-only and removal of V1–V3 attenuated the association.",
    "fig4_null_bootstrap": "Topology-level empirical nulls and uncertainty. (a–b) Null distributions from 1,000 joint time/event-pair permutations, refitting all 24 Cox models per draw; observed cosine and projection are marked. (c) Medians and percentile 95% intervals from 1,000 patient bootstraps.",
    "fig5_robustness": "Boundaries of whole-topology agreement. (a) Metrics under unadjusted, age-and-sex, and fully adjusted Cox models. (b) Cosine similarity for the original time-zero audit, local aVL Q–S-span adjustment, 0.5% feature-tail trimming, and corrected positive-follow-up primary analysis.",
    "figS1_subgroups": "Prespecified first-difference V1–V3 estimates in the normal-ECG subgroup (a) and after excluding major rhythm/conduction abnormalities (b). Points are hazard ratios per SD; bars are model-based 95% confidence intervals.",
    "figS2_sami": "Supportive disease-specific generalization from CODE discovery to SaMi-Trop. All 24 coefficients are retained; opposing effects are explicitly distinguished. SaMi-Trop outcomes had previously been inspected and this comparison is not confirmatory external validation.",
    "figS3_validity": "Corrected-replication aVL and V1–V3 estimates after adjustment for the local aVL Q–S-span proxy and after removing the outer 0.5% of each feature tail. Points are hazard ratios per SD; bars are model-based 95% confidence intervals.",
    "figS4_anatomy": "Anatomy-aware topology diagnostics. (a) Normalized graph-Laplacian energy in discovery and replication; lower values are smoother. (b) Anatomy-preserving discovery–replication alignment for each fixed feature family. Empirical p-values from 10,000 lead-label permutations are reported in the manuscript.",
    "figS5_schoenfeld": "Feature-specific rank-transformed Schoenfeld tests in discovery and corrected replication. The horizontal reference is the Bonferroni threshold 0.05/24; no feature term crossed it.",
    "figS6_waveforms": "Outcome-independent supportive waveform view in SaMi-Trop. Median aVL-aligned beats compare the lower and upper quintiles of the frozen CODE discovery score; ribbons show the interquartile range. Each record and lead was independently min–max normalized, matching the extractor's morphology scale. Score quintiles—not mortality outcomes—defined the groups.",
    "fig6_waveforms": "Outcome-independent supportive waveform view focused on aVL and V1–V3. Median aVL-aligned SaMi-Trop beats compare lower and upper quintiles of the frozen CODE discovery score; ribbons show the interquartile range. Score quintiles—not mortality outcomes—defined the groups.",
}


def main():
    fig1_topology()
    fig2_replication()
    fig3_transfer()
    fig4_null_bootstrap()
    fig5_robustness()
    fig6_waveforms_focal()
    main_names = [
        "fig1_topology", "fig2_replication", "fig3_frozen_transfer",
        "fig4_null_bootstrap", "fig5_robustness", "fig6_waveforms",
    ]
    pdfs = [f"{name}.pdf" for name in main_names]
    (OUT / "figure_manifest.json").write_text(json.dumps({
        "source": "saved machine-readable results; waveform grouping is outcome-independent",
        "pdf_files": pdfs,
        "captions_file": "CAPTIONS.md",
    }, indent=2) + "\n")
    caption_text = "# Figure captions\n\n" + "\n\n".join(
        f"## {name}\n\n{CAPTIONS[name]}" for name in main_names
    ) + "\n"
    (OUT / "CAPTIONS.md").write_text(caption_text)
    print("\n".join(pdfs))


if __name__ == "__main__":
    main()
