# Cohort audit

*Submission finalization: 2026-09-10 (America/Los_Angeles)*

## Frozen CODE discovery

- **28,934 patients / 1,037 deaths**, positive follow-up.
- Waveform parts 17, 0, and 1; one ECG per patient by stable lowest `exam_id` because acquisition timestamps are absent.
- The cohort was availability-restricted at freeze. The appropriate mortality-index waveform was unavailable for 12,964 otherwise eligible patients; 22 had genuinely missing outcomes.
- Discovery membership, feature definitions, adjustment variables, and saved coefficients were never refit or redefined. Reconstruction reproduced saved HRs to machine precision (maximum absolute difference 2.22e-16).

| Prespecified analysis | Patients | Deaths |
|---|---:|---:|
| All | 28,934 | 1,037 |
| Normal ECG | 11,686 | 176 |
| Major rhythm/conduction abnormalities excluded | 26,075 | 760 |

## Primary untouched CODE replication

- **14,447 patients / 536 deaths**, restricted to `timey > 0`.
- Eligible non-discovery patients in already processed parts 17, 0, 1, and 2.
- Same 24 features, within-cohort feature scaling, covariates, and subgroup definitions as discovery.

| Prespecified analysis | Patients | Deaths |
|---|---:|---:|
| All | 14,447 | 536 |
| Normal ECG | 5,453 | 81 |
| Major rhythm/conduction abnormalities excluded | 13,019 | 395 |

## Original replication audit

The initially saved result included ten deaths with `timey == 0`, yielding **14,457 patients / 546 deaths**. This conflicts with the frozen positive-follow-up rule and materially changes the topology: the original audit has Pearson 0.735, Spearman 0.522, cosine 0.843, and 19/24 signs, versus 0.550, 0.354, 0.776, and 20/24 in the corrected primary cohort. The original is preserved as an audit sensitivity, never primary inference.

## Supportive SaMi-Trop

- Public source: 1,631 Chagas-disease ECGs / 104 deaths.
- Ten extraction-QC failures, including one death, leave **1,621 / 103**.
- Same 24 features; adjustment uses age, sex, and ECG-predicted age because the CODE rhythm flags are unavailable.
- Outcomes were inspected during pipeline validation. Results are supportive disease-specific generalization only.

## Quarantined data and EchoNext

A legacy background job completed CODE part 3 during finalization. A persistent stop guard now prevents further downloading. Part 3 remains on disk but the cohort loader explicitly admits only parts 17/0/1/2; it does not enter any reported analysis. A partial part-4 archive is likewise unused. EchoNext was skipped because access and analysis-ready data were not immediately available.

## Machine-readable sources

- `results/ml4h_extension_manifest.json`
- `results/finalization_manifest.json`
- `results/final_cohort_characteristics.csv`
- `results/final_results_table.csv`
- `results/final_subgroup_results.csv`
- `results/topology_replication_metrics_corrected.csv`
- `results/cohort_reconstruction_checks.csv`
- `results/final_discovery_28934/cohort_restriction_audit.csv`
