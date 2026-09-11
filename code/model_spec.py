"""Frozen patient-level model specification used by the paper.

This module contains no cohort data. It is provided so reviewers can inspect or
rerun the exact feature-wise Cox, frozen-score, topology, and graph definitions
after obtaining the public source cohorts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FEATURES = [f"{family}_{lead}" for lead in LEADS for family in ("rs_diff", "rs_diff_2")]
RHYTHM_FLAGS = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"]
FULL_COVARIATES = ["age", "is_male", "nn_predicted_age", *RHYTHM_FLAGS]
PENALIZER = 0.01
BONFERRONI_ALPHA = 0.05 / 24


def active_covariates(frame: pd.DataFrame, requested=FULL_COVARIATES) -> list[str]:
    """Drop absent or constant adjustment variables, as done within subgroups."""
    return [name for name in requested if name in frame and frame[name].nunique(dropna=False) > 1]


def fit_feature(frame: pd.DataFrame, feature: str, extra_covariates=None) -> dict:
    """Fit one standardized feature in the prespecified penalized Cox model."""
    requested = [*FULL_COVARIATES, *(extra_covariates or [])]
    covariates = active_covariates(frame, requested)
    data = frame[["timey", "death", feature, *covariates]].dropna().copy()
    data[feature] = (data[feature] - data[feature].mean()) / data[feature].std(ddof=1)
    model = CoxPHFitter(penalizer=PENALIZER)
    model.fit(data, duration_col="timey", event_col="death", show_progress=False)
    summary = model.summary.loc[feature]
    return {
        "feature": feature,
        "n": len(data),
        "deaths": int(data.death.sum()),
        "log_hr": float(summary.coef),
        "hr_per_sd": float(np.exp(summary.coef)),
        "ci95_lower": float(np.exp(summary["coef lower 95%"])),
        "ci95_upper": float(np.exp(summary["coef upper 95%"])),
        "p_value": float(summary.p),
        "bonferroni_significant": bool(summary.p < BONFERRONI_ALPHA),
        "adjustment_covariates": ";".join(covariates),
    }


def fit_coefficient_map(frame: pd.DataFrame, extra_covariates=None) -> pd.DataFrame:
    return pd.DataFrame([fit_feature(frame, feature, extra_covariates) for feature in FEATURES])


def topology_metrics(discovery, replication) -> dict:
    """Compute the five complete-vector replication statistics."""
    discovery = np.asarray(discovery, dtype=float)
    replication = np.asarray(replication, dtype=float)
    return {
        "pearson_r": float(stats.pearsonr(discovery, replication).statistic),
        "spearman_rho": float(stats.spearmanr(discovery, replication).statistic),
        "cosine_similarity": float(np.dot(discovery, replication) / (np.linalg.norm(discovery) * np.linalg.norm(replication))),
        "sign_concordance": float(np.mean(np.sign(discovery) == np.sign(replication))),
        "projection": float(np.dot(discovery, replication) / np.linalg.norm(discovery)),
    }


def frozen_score(frame: pd.DataFrame, parameters: pd.DataFrame, selected_features=None) -> np.ndarray:
    """Apply discovery means, SDs, and coefficients without refitting weights."""
    selected = FEATURES if selected_features is None else list(selected_features)
    parameters = parameters.set_index("feature")
    score = np.zeros(len(frame), dtype=float)
    for feature in selected:
        row = parameters.loc[feature]
        score += row.discovery_beta * (frame[feature].to_numpy() - row.discovery_mean) / row.discovery_sd
    return score


def normalized_graph_energy(vector, edges, lead_order=LEADS) -> float:
    """Normalized graph-Laplacian energy for one 12-lead feature family."""
    vector = np.asarray(vector, dtype=float)
    index = {lead: position for position, lead in enumerate(lead_order)}
    numerator = sum((vector[index[a]] - vector[index[b]]) ** 2 for a, b in edges)
    return float(numerator / np.dot(vector, vector))

