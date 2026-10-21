# prevent-AD brain-measure prediction pipeline

Code for testing whether regional brain measures predict longitudinal cognitive and fluid-biomarker trajectories
in [PREVENT-AD cohort: confirm name and add reference]. Each outcome x brain-measure pair is modeled separately
with cross-validated regression and tested against a permutation null.

> **Status:** [analysis in progress / submitted / published: update]. This repository contains **code and job files only**:
> no participant data and no results (see *Data availability*). Fields in [brackets] need to be completed by the authors.

## Analysis overview

| Component | Description |
|---|---|
| Predictors | 6 brain measures, each with 200 cortical regions (regions named by network, e.g. `17networks_LH_DefaultC_PHC_1`; Schaefer-Yeo 17-network parcellation [confirm]): `T1T2` [T1w/T2w ratio: confirm], and resting-state functional connectivity from five seeds: entorhinal, perirhinal, hippocampal body, head and tail [confirm] |
| Outcomes | Person-level BLUP intercepts and slopes from longitudinal mixed models [describe model]. Set 1 (10 outcomes): plasma/CSF A-beta ratio and pTau [confirm], RBANS total, TMT time and errors. Set 2 (22 outcomes): RBANS attention, delayed memory, immediate memory, total scale and visuospatial/constructional indexes; Simoa A-beta 42/40 ratio and p-tau217; List A learning, immediate and delayed recall; TMT total time |
| Sample | [n participants per analysis: add; complete cases only, so it varies by outcome] |
| Models | **TabPFN** regressor (primary; `n_estimators=1`, `ignore_pretraining_limits=True`, `random_state=4`); **ridge** (`RidgeCV`, 50 log-spaced alphas from 1e-4 to 1e5, standardized features, alpha tuned inside every training fold); linear SVR (earlier runs) |
| Cross-validation | 4-fold `KFold(shuffle=True, random_state=4)`, identical for every model and outcome |
| Score | mean squared error across held-out folds (reported as negative MSE) |
| Significance | outcome-permutation test: 1000 permutations (2500 in a sensitivity run); p = (C + 1) / (n_perm + 1) |
| Feature selection | primary analysis: none (all 200 regions). Sensitivity: Random Forest (100 trees) keeps the top 20 or 25 regions, **selected inside each training fold only**. A variant that selects once on all data (`rf_fulldata`) leaks test information and is kept only to document the inflation; its p-values should not be interpreted |
| Multiple testing | Benjamini-Hochberg FDR across all tests (and within outcome); Bonferroni also reported |
| Interpretation | SHAP (permutation explainer, independent masker with 50 background participants) on held-out participants in each fold. SHAP describes what a fitted model relies on; it is not a significance test. Reported per region and aggregated per network |

## Repository layout

```
shared/        summarize_tabpfn_results.py   pairs_to_tasks.py   outcomes_new_blup.txt
tabpfn/        prevent-AD_sklearn_tabpfn_flexible.py    main TabPFN pipeline
  jobs/        LSF job files, one per analysis run (settings are at the top of each file)
  shap/        tabpfn_shap.py   summarize_tabpfn_shap.py   SHAP analysis and its job file
  archive/     earlier versions kept for reproducibility
regressions/   prevent-AD_sklearn_regressions_flexible.py   ridge (default) / linear SVR pipeline
  jobs/  archive/
xgboost/       earlier exploratory XGBoost script (not used for reported results)
environment/   package versions used
```

## How the pipeline runs

Jobs were run on an LSF cluster (University of Miami Pegasus [confirm]) as array jobs, one outcome x predictor pair per
task. Settings are passed by environment variables in the job files, for example:

| Variable | Meaning |
|---|---|
| `MODEL_TYPE` | `tabpfn`, `ridge` or `svr` |
| `N_PERM` | number of permutations |
| `FEATURE_SELECTION` | `none`, `rf_infold`, or `rf_fulldata` |
| `TOP_K` | regions kept when feature selection is on |
| `OUTCOME_LIST_FILE`, `OUTCOMES_FILE` | which outcomes to run and where to find them |
| `RESULTS_TAG` | suffix for the output folder |

Task number = (outcome position in `shared/outcomes_new_blup.txt` - 1) x 6 + predictor position (predictors in sorted
file-name order). Each task writes `*_results.txt` (score and permutation p), `*_permscores.npy` (null distribution)
and a histogram PNG. `shared/summarize_tabpfn_results.py <folder>` collects a folder into one table with FDR and
Bonferroni corrections and a p-value heatmap.

To run elsewhere, edit `data_dir` at the top of each script and the paths and queue settings in the job files.

## Software

Python 3.11, TabPFN 9.0.0, PyTorch 2.5.1, SHAP 0.51.0, scikit-learn, pandas, NumPy, seaborn
(exact versions in `environment/`). TabPFN requires its own access token and model terms; **no token is stored in
this repository** (job files contain `PASTE_TOKEN_HERE`).

## Data availability

The participant-level data are not shared here. [Describe how to request access, for example the data
governance body and contact.] The scripts expect one csv per brain measure (`*_400Parcels_bilateral.csv`, one row per
participant, a `subject` column and 200 region columns) and outcome files with a `subject` column.

## Limitations

- No covariates (age, sex, education, head motion) are included in the models yet.
- BLUP outcomes come from a mixed model fit on all participants before cross-validation (minor information sharing).
- Regional features are strongly correlated, so SHAP credit is shared somewhat arbitrarily among neighbors; network-level
  summaries are more stable than single-region rankings.

## Citation and license

[Add citation, DOI, funding, ethics statement and license once decided.]

Contact: [name, institutional email]
