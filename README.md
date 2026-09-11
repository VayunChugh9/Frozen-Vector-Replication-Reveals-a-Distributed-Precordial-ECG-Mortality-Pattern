# Anonymous ML4H reproduction repository

This repository contains a compact, code-first review package for auditing the reported cohorts, recomputing the main topology-level statistics, inspecting uncertainty and validity outputs, and regenerating Figures 1–6. The manuscript and pre-rendered figures are intentionally not distributed.

No raw ECGs, patient-level tables, patient or exam identifiers, credentials, author metadata, local paths, logs, caches, or Git history are included.

## What can be reproduced directly

From the saved machine-readable analysis outputs, a reviewer can reproduce:

- discovery and corrected-replication cohort counts;
- all five whole-topology similarity statistics;
- empirical permutation p-values from all 1,000 draws;
- bootstrap medians and percentile intervals from all 1,000 patient resamples;
- anatomy-label empirical p-values from 10,000 permutations per feature family;
- frozen-score transfer and prespecified ablation summaries;
- subgroup, morphology/duration, artifact-tail, and Schoenfeld validity summaries;
- all six figures used in the manuscript.

Run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python code/reproduce_main_results.py
python code/make_figures.py
```

The first command exits nonzero if a recomputed value differs from its saved reference. The figure command writes PDF and PNG versions to `figures/`.

## Repository layout

```text
anonymous_ml4h_repo/
├── README.md
├── requirements.txt
├── code/
│   ├── extract_terminal_qrs.py
│   ├── model_spec.py
│   ├── reproduce_main_results.py
│   └── make_figures.py
├── docs/
│   ├── ANALYSIS_PLAN.md
│   └── COHORT_AUDIT.md
├── results/
│   └── aggregate tables, null draws, bootstrap draws, and validity files
└── .gitignore
```

## Patient-level reproduction boundary

Raw ECGs and patient-level derived features are deliberately excluded from anonymous review. The exact frozen feature extraction and Cox/topology model definitions are provided in `code/extract_terminal_qrs.py` and `code/model_spec.py`. The cohort boundary and the material zero-follow-up correction are documented in `docs/ANALYSIS_PLAN.md` and `docs/COHORT_AUDIT.md`.

The primary inferential replication is the positive-follow-up cohort (14,447 patients, 536 deaths). The originally saved 14,457-patient/546-death analysis is retained only as a time-zero audit. Discovery remains frozen at 28,934 patients and 1,037 deaths.

`code/make_figures.py` creates a local `figures/` directory on demand. Generated graphics are intentionally ignored so the review package remains source-first.

## Included validity record

`results/unresolved_validity_concerns.csv` records the limitations that must remain explicit in any report based on these outputs. SaMi-Trop is supportive disease-specific generalization only; its previously inspected outcomes and opposing effects are not hidden.
