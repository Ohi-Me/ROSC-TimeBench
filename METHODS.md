# Methods

For the Correspondence "Elapsed time alone achieves event-level discrimination similar to a published ROSC prediction model, but not cycle-level discrimination".

## Datasets
1. **Zenodo** (doi:10.5281/zenodo.20162201; CC BY 4.0). Adult out-of-hospital cardiac arrests resuscitated in an emergency department.
   - Resuscitation duration: the recorded duration of emergency-department CPR, in minutes.
   - Outcome: ROSC (coded 1 = yes).
   - Baseline variables: age, sex, arrest location, bystander CPR, shockable rhythm and aetiology.
   - The file has 400 records; one with missing or non-positive duration was excluded, leaving 399.
2. **CAAM sub-study** (doi:10.17632/mbwykx585m.1; CC BY 4.0). Prehospital physician-staffed advanced-life-support resuscitations from the chest-compression-fraction sub-study of the CAAM trial.
   - Time origin: arrival of the advanced-life-support team.
   - Duration: minutes to first ROSC, or to the end of CPR without ROSC.
   - Clock times that crossed midnight were corrected by adding 24 hours.
   - Outcome: first ROSC recorded (yes/no).
   - Baseline variables: age, sex, witnessed arrest, bystander CPR and shockable rhythm.
   - All 112 records with a trial identifier were included.

Both datasets were analysed as released; no records were linked or re-identified.

## Prediction designs
- **Horizon.** Observation stopped at 30 minutes, matching Huang et al. ROSC after 30 minutes was counted as no ROSC within the horizon. A sensitivity analysis used no horizon.
- **Cycles.** Cycle *k* covers minutes (2(*k*−1), 2*k*]. A patient contributes cycles 1 to *C*, where *C* = ⌈min(duration, 30)/2⌉.
- **Cycle-level (landmark) prediction.**
  - When: at the start of each cycle (landmark *L* = 2(*k*−1) minutes), for every patient still being resuscitated without ROSC.
  - Target: ROSC during that cycle, i.e., within the next 2 minutes.
  - Available time information: the elapsed time *L*.
- **Event-level prediction.**
  - When: once per patient, at the start of the patient's last cycle.
  - Target: ROSC during that cycle, which equals ROSC within the horizon.
  - Time input: fixed by the completed resuscitation duration, which is known only after resuscitation has ended. It is therefore not available prospectively at earlier time points.

## Models
- **Method.** L2-penalised logistic regression (inverse penalty *C* = 0.3), with time entered as *L*/10 and (*L*/10)².
- **Event-level models**, fitted on one row per patient:
  - elapsed time only;
  - baseline variables only;
  - baseline variables plus elapsed time.
- **Cycle-level (landmark) models**, fitted on all cycles of the training patients:
  - elapsed time only;
  - baseline variables plus elapsed time.
- **Fixed-direction score.** "Elapsed time alone" is not a fitted model. Its score is −*L*, so shorter time means higher predicted ROSC, as at the event level. It is reported to show whether the event-level direction holds at the cycle level.
- **Transfer.** The event-level model with baseline variables plus elapsed time was also applied to every cycle of held-out patients. This shows what happens when a model trained at the event level is used at each cycle.

## Validation and uncertainty
- **Cross-validation.** 10-fold, with folds assigned by patient, so all cycles of one patient are in the same fold. Standardisation and mean imputation of missing baseline values were fitted on the training fold only.
- **Confidence intervals.** 95% intervals are percentile intervals from 2000 patient-level bootstrap resamples of the out-of-fold predictions; all cycles of a sampled patient were kept together. Differences between AUROCs use paired resamples. These intervals do not include model-refitting variability. That variability is reported separately as the SD across 20 repetitions of cross-validation with different fold assignments (`repeated_cv_sd` column of the results file).
- **Calibration.** Reported only for predictions and outcomes with the same target (ROSC within the next 2 minutes), as:
  - mean predicted versus observed probability;
  - calibration slope (logistic regression of the outcome on the logit of the prediction);
  - Brier score.

## Comparison with Huang et al.
Huang et al.'s results are taken from their publication. They come from a different cohort, predictors (sequential EtCO2) and model. They are shown as an external benchmark, not as a head-to-head comparison.

## Software
Python (numpy, pandas, scikit-learn, pyreadstat). `rosc_timebench.py` downloads both datasets and reproduces every result.

## Direction of elapsed time at the cycle level
Per-cycle ROSC rates by landmark window, 30-minute horizon:

| Landmark window | Zenodo: cycles, ROSC rate | CAAM: cycles, ROSC rate |
|---|---|---|
| 0–8 min | 1574, 1.4% | 431, 4.6% |
| 8–16 min | 1419, 3.2% | 335, 4.2% |
| 16–24 min | 1089, 3.7% | 225, 4.4% |
| 24–30 min | 632, 2.7% | 79, 2.5% |

Per-cycle ROSC rates did not fall with time over the first 24 minutes, and in the emergency-department cohort they rose. A fixed "shorter time = higher probability" score therefore has a cycle-level AUROC below 0.5. The code checks that ROSC is coded 1 = yes.

## Full results

All estimates, 95% CIs and repeated cross-validation SDs, for both cohorts with and without the 30-minute horizon, are in [`results/reference_results.csv`](results/reference_results.csv).
