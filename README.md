# prevent-AD: brain measures predicting longitudinal outcomes

Last updated: 2026-10-20  |  Owner: ixc751  |  See `RUNS.md` for one row per analysis run.

## What this project does

For each **outcome** (cognitive or fluid-biomarker BLUP intercept/slope) and each **brain measure**
(6 predictor files x 200 cortical regions), fit a model with 4-fold cross-validation, score it by
mean squared error, and test it against a permutation null. One model per outcome x predictor pair.

- Predictors (6): `T1T2`, `rsfc_entorhinal`, `rsfc_hipp_body`, `rsfc_hipp_head`, `rsfc_hipp_tail`, `rsfc_perirhinal`
- Outcomes: original set (10, `df_longitudinal_outcomes.csv`) and the new BLUP set (22, `outcomes_new_blup.txt`, found in `blup_outcomes_all.csv`)
- Models: TabPFN (main), ridge (comparison), earlier SVR/XGBoost runs

## Fixed decisions (do not change between runs unless you record why)

- Cross-validation: `KFold(n_splits=4, shuffle=True, random_state=4)`, identical for every model
- Score: negative MSE. Permutation p = (C + 1) / (n_perm + 1)
- Random seed: 4
- Feature selection: none (all 200 regions) unless the run folder says `rf_...`
- Valid feature selection = `rf_infold` (Random Forest ranks regions inside each training fold only).
  `rf_fulldata` selects on all subjects first, which leaks test information. **Do not report rf_fulldata p-values.**
- Multiple testing: report FDR (Benjamini-Hochberg). With 132 tests and 1000 permutations Bonferroni cannot
  be passed (smallest p = 0.001). Decide the correction family before looking at results.
- **Primary analysis (proposed, confirm with PI and edit here): TabPFN, all 200 regions, 1000 permutations
  (`tabpfn_newblup`). Everything else is a sensitivity check.**

## Locations

| What | Where |
|---|---|
| Raw data | `/projectnb/jamlab/prevent_AD_dataset/prevent_AD_dataset/` |
| Cluster working folder (scripts run here, results land here) | `/projectnb/jamlab/ixc751/` |
| Job logs | `/projectnb/jamlab/bsub_output/` (`out_<tag>.<jobid>.<task>`, `err_<tag>...`) |
| Conda environment | `/projectnb/jamlab/ixc751/conda_envs/prevent_AD` |
| Code, master copy (JupyterHub) | `/nethome/ixc751/prevent_AD/` (this folder). Edit here, then deploy to the cluster with `deploy_to_cluster.sh` |

Note: there are two different things called "tabpfn": the code folder in the home directory and the
results folder `/projectnb/jamlab/ixc751/tabpfn/`. Do not confuse them.

## Folder layout (JupyterHub master copy)

```
prevent_AD/
  README.md  RUNS.md  deploy_to_cluster.sh
  shared/        summarize_tabpfn_results.py  pairs_to_tasks.py  outcomes_new_blup.txt
  tabpfn/        prevent-AD_sklearn_tabpfn_flexible.py   (the one current script, v5)
    jobs/        prevent-AD_tabpfn_new_outcomes_cpu.job        all regions, 1000 perms, 22 new outcomes
                 prevent-AD_tabpfn_rf25_new_outcomes_cpu.job   RF top 25 in-fold, 22 new outcomes
                 prevent-AD_tabpfn_2500perm_new_outcomes_cpu.job  all regions, 2500 perms
                 prevent-AD_tabpfn_rf_cpu.job / _rf_gpu.job    RF top 20 in-fold, 10 old outcomes
    shap/        tabpfn_shap.py  summarize_tabpfn_shap.py  prevent-AD_tabpfn_shap_gpu.job
    archive/     v4 script, 1 Oct script, first job files
  regressions/   prevent-AD_sklearn_regressions_flexible.py  (ridge v2; also LinearSVR via MODEL_TYPE)
    jobs/        prevent-AD_ridge_new_outcomes.job
    archive/     old ridge/SVR job and script versions
  xgboost/       prevent-AD_sklearn_xgboost_flexible.py   (older, inactive)
    jobs/  archive/
  results/       shortcuts and exports (symlinks to /projectnb result folders plus copied plots):
                 tabpfn_results  results_rf_infold  regressions_results  xgBoost_results
                 tabpfn_plots  results_to_download
    archive/
  archive/       cross-model leftovers: old summarizer copies, early summary CSVs
```

The cluster working folder (`/projectnb/jamlab/ixc751/`) is **flat**: jobs `cd` there and call scripts by plain
name, so deployed copies must keep those names. Never move or rename scripts there while jobs are queued.

Result folders are named by their settings: `<model>_<selection>_<topK>_<outcome set>_<tag>`, for example
`tabpfn_rf_infold_top25_newblup`. Add a tag when a setting changes (`_2500perm`).

## Program inventory

| File | What it does | Reads | Writes |
|---|---|---|---|
| `prevent-AD_sklearn_tabpfn_flexible.py` (v5) | TabPFN + permutation test. Options by environment variable: `N_PERM`, `FEATURE_SELECTION` (none / rf_infold / rf_fulldata), `TOP_K`, `OUTCOME_LIST_FILE`, `OUTCOMES_FILE` (`auto` searches the data folder), `RESULTS_TAG`. `LSB_JOBINDEX` picks one pair. | data folder | `<folder>/*_results.txt`, `*_permscores.npy`, `*.png`, `*_selected_features.csv` (RF runs) |
| `prevent-AD_sklearn_regressions_flexible.py` (ridge v2) | RidgeCV (alpha re-tuned inside each fold) or LinearSVR, same CV and permutation test, same output files. `MODEL_TYPE=ridge|svr`. | data folder | `ridge_<tag>/...` |
| `tabpfn_shap.py` | SHAP importance for chosen pairs (same 4 folds, explains held-out people). `SHAP_PAIRS`, `SHAP_FEATURES`. Descriptive only, not a significance test. | data folder | `shap_tabpfn_all/*_regions.csv`, `_networks.csv`, `_beeswarm.png`, `_values.npz` |
| `summarize_tabpfn_results.py` | Collects a results folder into one table with FDR and Bonferroni plus a p-value heatmap. `EXPECTED_PAIRS`, `REQUIRED_N_PERM` environment variables. | `<folder>/*_results.txt` | `summary_tabpfn_results.csv`, `summary_tabpfn_pvalue_heatmap.png` (inside the folder) |
| `summarize_tabpfn_shap.py` | Combines SHAP outputs: top regions per pair, network and region matrices, consistency across pairs, heatmaps, Excel workbook. | `shap_tabpfn_all/` | `shap_tabpfn_all/summary/` |
| `pairs_to_tasks.py` | Turns significant pairs from a summary CSV into array task numbers for reruns. | summary CSV | printed list |
| `outcomes_new_blup.txt` | The 22 new outcome names, in order (order sets task numbers). | | |

## How to run an analysis

1. Edit files only in the JupyterHub master copy. Then, in the Pegasus terminal: `bash /nethome/ixc751/prevent_AD/deploy_to_cluster.sh`
   (copies scripts, jobs and the outcome list to `/projectnb/jamlab/ixc751/`, fixes line endings, prints script versions,
   and lists unmanaged files in the cluster folder as candidates for `_archive/`).
2. Token: stored once in `/projectnb/jamlab/ixc751/.tabpfn_env` (`chmod 600`). The deploy script inserts it into job
   files; master job files keep `PASTE_TOKEN_HERE`. Never paste the token into notes or chat.
3. Test one task: `bsub -J "name[1]" < file.job`, then `bpeek <jobid>[1]` and check the ETA.
4. Submit the rest: `bsub -J "name[2-132]" < file.job`.
5. Check completion: `grep -l "Number of permutations: N" <folder>/*_results.txt | wc -l`.
6. Summarize: `EXPECTED_PAIRS=132 python summarize_tabpfn_results.py <folder>`.
7. Copy summaries to JupyterHub with the folder name in the file name so nothing is overwritten.
8. Add a row to `RUNS.md`.

### Array task numbers (new outcomes, 132 pairs)

task = (outcome position in `outcomes_new_blup.txt` - 1) x 6 + predictor position (1-based), predictors in
the order listed above. Example: task 18 = 3rd outcome x `rsfc_perirhinal`.

## Known issues and cautions

- Node `n078` gave `Illegal instruction` (exit code 132) on one task. Resubmit the task; exclude the node
  with `-R "span[hosts=1] rusage[mem=2G] select[hname!=n078]"` if it repeats.
- LSF shows a job as "Done" even if Python crashed, unless the job file ends with `exit $status`. Current jobs do.
- No covariates (age, sex, education, head motion) are included in any model yet.
- BLUP outcomes come from a mixed model fit on all subjects before cross-validation (minor leakage, mention in methods).
- Orbitofrontal and anterior temporal rsfc has weaker signal (susceptibility artifact); interpret limbic-network SHAP with care.
- The three AB_ratio_slope hippocampal-seed models are not independent replications.
- SHAP says what a model used, not where pathology "lives". Use "predicts", not "drives".
- The TabPFN token is in the job files. Regenerate it if the files were ever shared.

## Notes on specific folders (fill in the blanks)

| Folder | What it is |
|---|---|
| `results/regressions_results` | early ridge/SVR plots (was `regression/`) (check) |
| `results/xgBoost_results`, `xgboost/` | older XGBoost runs; inactive (fill in what they predicted) |
| `results/tabpfn_plots`, `archive/tabpfn_summary_results.csv` | early-October exports (which run?) |
| `results/results_to_download` | copies for downloading; keep unique names |
| `/projectnb/jamlab/ixc751/_archive/` | cluster files moved out of the working folder (never deleted) |

## Cleanup rules

- Replace a file, archive the old one first: `mv -n file archive/file_v<N>_<date>`.
- Never `rm` code or results during the project. Use `mv -n` into an `archive/` or `_archive/` folder.
- No `(1)`, `(2)` copies in the master folders. Rename the newest upload to the plain name.
