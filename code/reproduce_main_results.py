"""Recompute and validate the main-paper statistics from included outputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FEATURES = [f"{family}_{lead}" for lead in LEADS for family in ("rs_diff", "rs_diff_2")]
TOLERANCE = 1e-10


def close(actual, expected, label):
    if not np.isclose(actual, expected, atol=TOLERANCE, rtol=TOLERANCE):
        raise AssertionError(f"{label}: recomputed={actual!r}, saved={expected!r}")


def topology_checks():
    coefficients = pd.read_csv(RESULTS / "adjustment_robustness_coefficients.csv")
    coefficients = coefficients.query("specification == 'full'")
    discovery = coefficients.query("cohort == 'CODE_discovery'").set_index("feature").loc[FEATURES, "log_hr"].to_numpy()
    replication = coefficients.query("cohort == 'CODE_replication_corrected'").set_index("feature").loc[FEATURES, "log_hr"].to_numpy()
    calculated = {
        "pearson_r": stats.pearsonr(discovery, replication).statistic,
        "spearman_rho": stats.spearmanr(discovery, replication).statistic,
        "cosine_similarity": np.dot(discovery, replication) / (np.linalg.norm(discovery) * np.linalg.norm(replication)),
        "sign_concordance": np.mean(np.sign(discovery) == np.sign(replication)),
        "projection": np.dot(discovery, replication) / np.linalg.norm(discovery),
    }
    saved = pd.read_csv(RESULTS / "topology_replication_metrics_corrected.csv").iloc[0]
    for key, value in calculated.items():
        close(value, saved[key], f"topology.{key}")
    return calculated


def cohort_checks():
    table = pd.read_csv(RESULTS / "final_results_table.csv")
    expected = {
        ("CODE_discovery", "all"): (28934, 1037),
        ("CODE_replication_corrected", "all"): (14447, 536),
        ("CODE_replication_original_audit", "all"): (14457, 546),
        ("SaMi_Trop_supportive", "all"): (1621, 103),
        ("CODE_discovery", "normal_ecg"): (11686, 176),
        ("CODE_discovery", "rhythm_abnormalities_excluded"): (26075, 760),
        ("CODE_replication_corrected", "normal_ecg"): (5453, 81),
        ("CODE_replication_corrected", "rhythm_abnormalities_excluded"): (13019, 395),
    }
    for key, target in expected.items():
        rows = table.query("cohort == @key[0] and subgroup == @key[1]")
        if len(rows) != 24:
            raise AssertionError(f"{key}: expected 24 features, found {len(rows)}")
        observed = (int(rows.n.iloc[0]), int(rows.deaths.iloc[0]))
        if observed != target:
            raise AssertionError(f"{key}: observed={observed}, expected={target}")
    return {"/".join(key): value for key, value in expected.items()}


def permutation_checks():
    draws = pd.read_csv(RESULTS / "topology_outcome_permutation_draws.csv")
    summary = pd.read_csv(RESULTS / "topology_outcome_permutation_summary.csv").set_index("statistic")
    if len(draws) != 1000:
        raise AssertionError(f"Expected 1,000 permutation draws, found {len(draws)}")
    output = {}
    for statistic in ["cosine_similarity", "projection"]:
        observed = float(summary.loc[statistic, "observed"])
        empirical_p = (1 + int((draws[statistic] >= observed).sum())) / (len(draws) + 1)
        close(empirical_p, summary.loc[statistic, "empirical_p"], f"permutation.{statistic}.p")
        output[statistic] = {"observed": observed, "empirical_p": empirical_p}
    return output


def bootstrap_checks():
    draws = pd.read_csv(RESULTS / "replication_bootstrap_draws.csv")
    summary = pd.read_csv(RESULTS / "replication_bootstrap_summary.csv").set_index("statistic")
    if len(draws) != 1000:
        raise AssertionError(f"Expected 1,000 bootstrap draws, found {len(draws)}")
    output = {}
    for statistic in ["pearson_r", "spearman_rho", "cosine_similarity", "projection", "sign_concordance"]:
        quantiles = draws[statistic].quantile([0.025, 0.5, 0.975]).to_numpy()
        saved = summary.loc[statistic, ["lower_95", "median", "upper_95"]].to_numpy(dtype=float)
        if not np.allclose(quantiles, saved, atol=TOLERANCE, rtol=TOLERANCE):
            raise AssertionError(f"bootstrap.{statistic}: recomputed={quantiles}, saved={saved}")
        output[statistic] = quantiles.tolist()
    # Match the saved pipeline: take log-HR quantiles, then exponentiate.
    score_quantiles = np.exp(draws.frozen_score_log_hr.quantile([0.025, 0.5, 0.975]).to_numpy())
    score_saved = summary.loc["frozen_score_log_hr", ["lower_95", "median", "upper_95"]].to_numpy(dtype=float)
    if not np.allclose(score_quantiles, score_saved, atol=TOLERANCE, rtol=TOLERANCE):
        raise AssertionError("bootstrap.frozen_score_hr differs from saved summary")
    output["frozen_score_hr"] = score_quantiles.tolist()
    return output


def anatomy_checks():
    null = pd.read_parquet(RESULTS / "anatomy_label_permutation_null.parquet")
    saved = pd.read_csv(RESULTS / "anatomy_topology_results.csv").set_index("feature_family")
    output = {}
    for family in ["rs_diff", "rs_diff_2"]:
        draws = null.query("feature_family == @family")
        if len(draws) != 10000:
            raise AssertionError(f"{family}: expected 10,000 anatomy draws, found {len(draws)}")
        row = saved.loc[family]
        calculated = {
            "discovery_smoothness_empirical_p": (1 + int((draws.discovery_energy <= row.discovery_energy).sum())) / 10001,
            "replication_smoothness_empirical_p": (1 + int((draws.replication_energy <= row.replication_energy).sum())) / 10001,
            "alignment_empirical_p": (1 + int((draws.alignment_cosine >= row.alignment_cosine).sum())) / 10001,
        }
        for key, value in calculated.items():
            close(value, row[key], f"anatomy.{family}.{key}")
        output[family] = calculated
    return output


def validity_checks():
    ph = pd.read_csv(RESULTS / "schoenfeld_ph_tests.csv")
    if len(ph) != 48 or int(ph.feature_ph_bonferroni_significant.sum()) != 0:
        raise AssertionError("Feature-specific Schoenfeld result does not match 0/48 violations")
    transfer = pd.read_csv(RESULTS / "frozen_score_transfer.csv")
    required_scores = {"full_12lead", "precordial_only", "limb_only", "full_minus_aVL", "full_minus_V1_V3"}
    corrected = transfer.query("cohort == 'CODE_replication_corrected'")
    if not required_scores.issubset(set(corrected.score)):
        raise AssertionError("Frozen-score ablation block is incomplete")
    waveform = pd.read_csv(RESULTS / "sami_frozen_score_waveform_summary.csv")
    if set(waveform.lead) != set(LEADS) or set(waveform.score_group) != {"Lower quintile", "Upper quintile"}:
        raise AssertionError("Waveform summary lacks a lead or frozen-score group")
    return {
        "schoenfeld_feature_violations": 0,
        "corrected_ablation_scores": sorted(required_scores),
        "waveform_summary_rows": len(waveform),
    }


def main():
    report = {
        "cohorts": cohort_checks(),
        "topology": topology_checks(),
        "permutation": permutation_checks(),
        "bootstrap": bootstrap_checks(),
        "anatomy": anatomy_checks(),
        "validity": validity_checks(),
    }
    print(json.dumps({"status": "PASS", **report}, indent=2))


if __name__ == "__main__":
    main()
