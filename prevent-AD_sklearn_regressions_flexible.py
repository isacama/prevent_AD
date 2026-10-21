"""
Ridge (default) or LinearSVR regression + permutation test for prevent-AD.
Mirrors the TabPFN pipeline so the two can be compared directly:
  * same 4-fold CV (KFold, shuffle=True, random_state=4)
  * same score (negative MSE) and 1000-permutation p-value
  * same results files (*_results.txt, *_permscores.npy, *.png) so that
    summarize_tabpfn_results.py works on the output folder unchanged

Settings come from environment variables (set them in the job file):
  MODEL_TYPE         ridge (default) | svr
  N_PERM             permutations (default 1000)
  N_JOBS             parallel workers for the permutations (default 8)
  OUTCOME_LIST_FILE  text file, one outcome column per line ("" = all columns of the
                     outcomes file, the original behaviour)
  OUTCOMES_FILE      csv in data_dir ("df_longitudinal_outcomes.csv" by default),
                     several comma-separated, or "auto" = search every csv for the columns
  RESULTS_TAG        appended to the output folder name (ridge_<tag>/)
  SKIP_DONE          1 (default) = skip pairs whose results file already exists
Array jobs: LSB_JOBINDEX picks one outcome x predictor pair (1-based); without it
all pairs run one after another.
"""
import glob
import os
import sys
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.experimental import enable_halving_search_cv  # noqa: F401  (needed for Halving*)
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import HalvingGridSearchCV, KFold, permutation_test_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

warnings.filterwarnings("ignore")
np.set_printoptions(precision=3)

SCRIPT_VERSION = "2026-10-14 ridge v2 (array jobs, outcome list, results files)"
data_dir = os.getenv("DATA_DIR", "/projectnb/jamlab/prevent_AD_dataset/prevent_AD_dataset/")

MODEL_TYPE = os.getenv("MODEL_TYPE", "ridge")
N_PERM = int(os.getenv("N_PERM", "1000"))
N_JOBS = int(os.getenv("N_JOBS", "8"))
OUTCOMES_FILE = os.getenv("OUTCOMES_FILE", "df_longitudinal_outcomes.csv")
OUTCOME_LIST_FILE = os.getenv("OUTCOME_LIST_FILE", "")
RESULTS_TAG = os.getenv("RESULTS_TAG", "")
SKIP_DONE = os.getenv("SKIP_DONE", "1") == "1"
SEED = 4


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------
# Outcome loading (same logic as the TabPFN script)
# ----------------------------------------------------------------------
def read_table(path):
    first = pd.read_csv(path, nrows=0).columns[0]
    return pd.read_csv(path, index_col=None if first == "subject" else 0)


def load_outcomes():
    wanted = None
    if OUTCOME_LIST_FILE:
        with open(OUTCOME_LIST_FILE) as fh:
            wanted = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
        wanted = list(dict.fromkeys(wanted))

    if OUTCOMES_FILE == "auto":
        if not wanted:
            raise ValueError("OUTCOMES_FILE=auto needs OUTCOME_LIST_FILE")
        files, remaining = [], set(wanted)
        for path in sorted(glob.glob(data_dir + "*.csv")):
            if path.endswith("_400Parcels_bilateral.csv"):
                continue
            try:
                hdr = set(read_table(path).columns)
            except Exception:
                continue
            if "subject" in hdr and (remaining & hdr):
                files.append(os.path.basename(path))
                log(f"  {os.path.basename(path)} supplies {len(remaining & hdr)} outcome(s)")
                remaining -= hdr
    else:
        files = [f.strip() for f in OUTCOMES_FILE.split(",") if f.strip()]

    merged = None
    for fn in files:
        d = read_table(data_dir + fn)
        if "subject" not in d.columns:
            raise ValueError(f"No 'subject' column in {fn}")
        have = set() if merged is None else set(merged.columns)
        cols = [c for c in d.columns if c != "subject" and c not in have
                and (wanted is None or c in wanted)]
        d = d[["subject"] + cols]
        merged = d if merged is None else pd.merge(merged, d, on="subject", how="outer")

    if wanted is None:
        outcomes = merged.columns[1:].tolist()
    else:
        outcomes = [w for w in wanted if w in merged.columns]
        missing = [w for w in wanted if w not in merged.columns]
        if missing:
            log(f"ERROR: {len(missing)} requested outcome(s) not found: {missing}")
            raise SystemExit(1)
    return merged[["subject"] + outcomes], outcomes


# ----------------------------------------------------------------------
def build_model(model_type):
    if model_type == "ridge":
        # alpha is re-tuned inside every training fold (nested), so the CV score is honest
        regression = RidgeCV(alphas=np.logspace(-4, 5, 50), scoring="neg_mean_squared_error")
        return make_pipeline(StandardScaler(), regression)
    if model_type == "svr":
        pipeline = Pipeline([("scaler", StandardScaler()),
                             ("svr", LinearSVR(dual="auto", tol=1e-3))])
        param_grid = {"svr__C": [0.01, 0.1, 1.0, 10],
                      "svr__epsilon": [0.001, 0.01, 0.1, 0.5, 0.75]}
        return HalvingGridSearchCV(pipeline, param_grid, cv=KFold(n_splits=4, shuffle=False),
                                   scoring="neg_mean_squared_error")
    raise ValueError(f"Unknown MODEL_TYPE: {model_type}")


def predict_longitudinal_outcome(pred_path, df_outcomes, outcome_var, model_type, n_perm):
    t_start = time.time()
    predictor_name = os.path.basename(pred_path).split(".")[0]
    output_dir = model_type + (f"_{RESULTS_TAG}" if RESULTS_TAG else "")
    os.makedirs(output_dir, exist_ok=True)
    tag = f"baseline_{predictor_name}_{model_type}_{outcome_var}"
    results_path = f"{output_dir}/{tag}_results.txt"

    if SKIP_DONE and os.path.exists(results_path):
        log(f"Already done, skipping: {results_path}")
        return

    log(f"\nOutcome: {outcome_var}\nPredictor: {predictor_name}")
    df_pred = pd.read_csv(pred_path, index_col=0)
    df_model = pd.merge(df_outcomes[["subject", outcome_var]], df_pred,
                        on="subject", how="outer").dropna(how="any")
    log(f"Dataset shape: {df_model.shape}")

    X = np.asfortranarray(df_model.iloc[:, 2:], dtype=np.float64)
    y = np.asfortranarray(df_model[outcome_var])
    final_model = build_model(model_type)

    score, perm_scores, pvalue = permutation_test_score(
        estimator=final_model, X=X, y=y, scoring="neg_mean_squared_error",
        cv=KFold(n_splits=4, shuffle=True, random_state=SEED),
        n_permutations=n_perm, n_jobs=N_JOBS, verbose=0, random_state=SEED)
    log(f"Original score (neg MSE): {score:.4f}   permutation p = {pvalue:.4f}")

    # plot
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.histplot(perm_scores, ax=ax)
    ax.axvline(score, color="red")
    ax.set_title(f"{predictor_name.replace('_400Parcels_bilateral', '')} -> {outcome_var}",
                 fontsize=7)
    fig.savefig(f"{output_dir}/{tag}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    np.save(f"{output_dir}/{tag}_permscores.npy", perm_scores)

    # chosen hyperparameters on the full data (descriptive)
    final_model.fit(X, y)
    if model_type == "ridge":
        best = f"alpha = {final_model.named_steps['ridgecv'].alpha_:.4g}"
    else:
        best = str(final_model.best_params_)
    log(f"Best parameters on full data: {best}")

    with open(results_path, "w") as f:
        f.write(f"Predictor: {predictor_name}\n")
        f.write(f"Outcome: {outcome_var}\n")
        f.write(f"Model: {model_type}\n")
        f.write(f"Dataset Shape: {df_model.shape}\n")
        f.write(f"Number of permutations: {n_perm}\n")
        f.write(f"Original MSE Score: {score}\n")
        f.write(f"Permutation p-value: {pvalue}\n")
        f.write(f"Best parameters (full data): {best}\n")
        f.write(f"Runtime (min): {(time.time() - t_start) / 60:.2f}\n")
    log(f"Saved {results_path}  ({(time.time() - t_start) / 60:.1f} min)")


def main():
    log(f"Script version: {SCRIPT_VERSION}")
    log(f"MODEL_TYPE={MODEL_TYPE}  N_PERM={N_PERM}  N_JOBS={N_JOBS}")
    log(f"OUTCOMES_FILE={OUTCOMES_FILE}  OUTCOME_LIST_FILE={OUTCOME_LIST_FILE or '(none: all columns)'}"
        f"  RESULTS_TAG={RESULTS_TAG or '(none)'}")
    df_outcomes, outcomes = load_outcomes()
    predictors = sorted(glob.glob(data_dir + "*_400Parcels_bilateral.csv"))
    all_pairs = [(o, p) for o in outcomes for p in predictors]
    log(f"Total outcome x predictor pairs: {len(all_pairs)}")

    idx = os.getenv("LSB_JOBINDEX")
    if idx is not None and int(idx) > 0:
        if int(idx) > len(all_pairs):
            log(f"Index {idx} out of range for {len(all_pairs)} pairs. Exiting.")
            return
        all_pairs = [all_pairs[int(idx) - 1]]
        log(f"--- Array task {idx}: {all_pairs[0][0]} x {os.path.basename(all_pairs[0][1])} ---")
    for outcome, pred in all_pairs:
        predict_longitudinal_outcome(pred, df_outcomes, outcome, MODEL_TYPE, N_PERM)


if __name__ == "__main__":
    main()
