# Anonymous ML4H reproduction repository

This repository contains a compact, code-first review package for auditing the reported cohorts, recomputing the main topology-level statistics, inspecting uncertainty and validity outputs, and regenerating figures.

## What can be reproduced directly

Run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python code/reproduce_main_results.py
python code/make_figures.py
```

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

The exact frozen feature extraction and Cox/topology model definitions are provided in `code/extract_terminal_qrs.py` and `code/model_spec.py`. The cohort boundary and the material zero-follow-up correction are documented in `docs/ANALYSIS_PLAN.md` and `docs/COHORT_AUDIT.md`.

The primary inferential replication is the positive-follow-up cohort (14,447 patients, 536 deaths). The originally saved 14,457-patient/546-death analysis is retained only as a time-zero audit. Discovery remains frozen at 28,934 patients and 1,037 deaths.

`code/make_figures.py` creates a local `figures/` directory on demand.

## Included validity record

`results/unresolved_validity_concerns.csv` records the limitations that must remain explicit in any report based on these outputs. SaMi-Trop is supportive disease-specific generalization only; its previously inspected outcomes and opposing effects are not hidden.
