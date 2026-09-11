# Frozen discovery / replication plan — CODE-15% terminal-QRS topology

**Frozen:** 2026-09-11, before opening any results from subsequently completed CODE
shards.  This supersedes the planned size-triggered discovery freeze.

## Amendment — availability-restricted final discovery cohort

For computational/time feasibility, final discovery is restricted to patients with a valid
mortality-index ECG waveform and nonmissing `death`/`timey` already available in the
processed shards. This is an availability restriction made without using biomarker effect
estimates. The 12,964 patients whose proper mortality-index waveform remains in an
unprocessed shard, and the 22 genuinely outcome-missing patients, are excluded from
discovery. All subsequently processed patients, including those with later-available
proper index waveforms, are internal replication only.

## Discovery cohort (frozen)

- Final outcome-eligible patients: **28,934**
- Final deaths: **1,037**
- Source feature shards: 17, 0, 1 only
- One ECG per patient: stable sort by `patient_id`, then lowest `exam_id`. The supplied
  metadata has no acquisition-time field, so this is a deterministic, outcome-blind
  fallback.
- Exclusions: unmatched exam IDs, `qc_fail_any`, non-positive follow-up, and missing
  outcome/follow-up. No ID or outcome value is forced or imputed.

The 24 discovery-model rows and the two pre-specified subgroup analyses are final for this
availability-restricted discovery cohort and will not be re-fit with later CODE shards.

## Replication hypothesis (frozen)

The target is a **spatial terminal-QRS mortality-risk topology** across all 12 leads and
both frozen features (`rs_diff`, `rs_diff_2`), assessed with the same extraction, QC,
one-ECG-per-patient rule, outcome definition, and fixed Cox covariate strategy used for
discovery. V1–V3 `rs_diff` are an a priori focal region for reporting topology concordance,
not a basis for omitting, re-weighting, or selecting the other 21 tests.

## Internal replication

All subsequently processed CODE patients are replication-only. No feature definition,
lead set, covariate, QC threshold, subgroup, or model parameter may be changed after
viewing replication results.

For all 24 fixed models, report per-SD HR, 95% CI, p-value, concordance, direction,
lead-ranking concordance with discovery, and formal discovery-vs-replication heterogeneity.
Null and discordant results are retained.

## External validation

SaMi-Trop uses the identical 12-lead extraction and fixed feature list. The Cox adjustment
strategy is age, sex, ECG-predicted age, and conventional ECG flags wherever the variable
exists and has non-zero variance. Any unavailable or subgroup-constant covariate is omitted
only for that stated numerical/data-availability reason and is recorded in the output.

## Outcome-missingness linkage

The 12,964 patients with a metadata-labelled outcome ECG but unavailable proper waveform
are not added to discovery. They remain eligible only for the replication workflow when
their waveform shard is processed. The 22 genuinely outcome-missing patients are excluded.
