# ROSC-TimeBench

How much does elapsed resuscitation time alone contribute to event-level versus cycle-level prediction of return of spontaneous circulation (ROSC)?

Analysis code for the Correspondence on Huang et al., *Resuscitation* 2026;229:111387.

Rohit Kumar, Urvashi Bansal. Centre for Artificial Intelligence, Dr. B. R. Ambedkar National Institute of Technology Jalandhar, India.

## Designs
- **Event level.** One prediction per resuscitation, at its last 2-minute cycle. The time input is the completed resuscitation duration.
- **Cycle level (landmark).** One prediction at the start of every 2-minute cycle, for ROSC within the next 2 minutes. The time input is the elapsed time at that moment.

Both designs use a 30-minute horizon. Definitions, models, validation and uncertainty estimation are described in [METHODS.md](METHODS.md).

## Data
Both datasets are downloaded automatically and are licensed CC BY 4.0.
- Jaradat Ö, Şahin B, Caliskan HM, Küçükceran K, Kılıç A. *The role of prior healthcare utilization and comorbidity burden in predicting return of spontaneous circulation in out-of-hospital cardiac arrest* [dataset]. Zenodo 2026. doi:10.5281/zenodo.20162201
- Malinverni S, Bartiaux M, Cavallotto F, et al. *Chest compression fraction during out of hospital cardiac arrest. Substudy of the CAAM trial* [dataset]. Mendeley Data 2018. doi:10.17632/mbwykx585m.1

## Run
```
pip install -r requirements.txt
python rosc_timebench.py --data data --out results.csv
```
If Mendeley Data cannot be reached, download the `.dta` file from https://data.mendeley.com/datasets/mbwykx585m/1 into `data/`, with a name starting `mbwykx585m_`, and rerun. The expected output is `results/reference_results.csv`.

## Main results
30-minute horizon; AUROC (95% CI).

| | Zenodo, emergency department (n=399) | CAAM, prehospital ALS (n=112) |
|---|---|---|
| Event level: elapsed time alone | 0.872 (0.832–0.907) | 0.891 (0.822–0.947) |
| Cycle level: elapsed time alone, shorter = higher | 0.413 (0.372–0.454) | 0.487 (0.413–0.562) |
| Cycle level: fitted elapsed-time model | 0.583 (0.538–0.629) | 0.452 (0.375–0.529) |
| Cycle level: baseline + time landmark model | 0.689 (0.642–0.734) | 0.572 (0.487–0.658) |
| Event-level model at every cycle: mean predicted vs observed | 0.612 vs 0.026 | 0.641 vs 0.043 |

- **Cycle-level AUROCs below 0.5.** Per-cycle ROSC rates do not fall over the first 24 minutes; in the emergency-department cohort they rise from 1.4% to 3.7%.
- **High cycle-level predictions from the event-level model.** They reflect a different target population (final cycles, ROSC 31–41%).

## Limitations
- Small cohorts, different from that of Huang et al.
- No EtCO2 measurements.
- Huang et al.'s figures are an external reference, not a head-to-head comparison.

## Licence
Code: MIT. The datasets keep their CC BY 4.0 licences; cite them as above.
