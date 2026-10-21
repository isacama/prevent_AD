import os

# ----------------------------------------------------------------------
# Environment / cache locations (set BEFORE importing torch / tabpfn).
# setdefault: the job file's exports win; these are fallbacks for
# interactive runs.
# ----------------------------------------------------------------------
PROJECT_DIR = "/projectnb/jamlab/ixc751"
os.environ.setdefault("HOME", PROJECT_DIR)
os.environ.setdefault("XDG_CACHE_HOME", f"{PROJECT_DIR}/.cache")
os.environ.setdefault("TABPFN_CACHE_DIR", f"{PROJECT_DIR}/.cache/tabpfn")
os.environ.setdefault("HF_HOME", f"{PROJECT_DIR}/.cache/huggingface")
os.environ.setdefault("TORCH_HOME", f"{PROJECT_DIR}/.cache/torch")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.makedirs(os.environ["TABPFN_CACHE_DIR"], exist_ok=True)

import glob
import inspect
import sys
import time
import warnings

import matplotlib

matplotlib.use("Agg")  # no display on compute nodes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    KFold,
    permutation_test_score,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor
from tabpfn import TabPFNRegressor
import tabpfn

warnings.filterwarnings("ignore")
np.set_printoptions(precision=3)

SCRIPT_VERSION = "2026-10-14 v5 (v4 + custom outcome list / outcome files / results tag)"
data_dir = "/projectnb/jamlab/prevent_AD_dataset/prevent_AD_dataset/"

# ----------------------------------------------------------------------
# Settings (override from the job file; no need to edit this script)
# ----------------------------------------------------------------------
MODEL_TYPE = os.getenv("MODEL_TYPE", "tabpfn")
N_PERM = int(os.getenv("N_PERM", "1000"))  # 100 = screening; 1000 for final runs

# FEATURE_SELECTION (TabPFN only):
#   none        -> use all features (what the 10/1 runs did)
#   rf_infold   -> RECOMMENDED. Random Forest ranks features using ONLY the
#                  training fold, keeps the top K, TabPFN fits on those.
#                  Redone in every fold of every permutation, so the test
#                  subjects never influence which features are picked.
#   rf_fulldata -> Same RF ranking, but done ONCE on all subjects with the
#                  real outcome; the same K features are then used for the
#                  whole CV + permutation test. Matches "pick top 20, then
#                  rerun", but the score/p-value are optimistically biased
#                  (test subjects helped choose the features).
# Which outcomes to run (defaults reproduce the original 10-outcome analysis):
#   OUTCOME_LIST_FILE  text file, one outcome column name per line ("" = all columns of
#                      the outcomes file, the original behaviour)
#   OUTCOMES_FILE      csv in data_dir holding the outcomes. Comma-separate several files,
#                      or use "auto" to search every csv in data_dir for the listed columns
#   RESULTS_TAG        appended to the output folder name so new runs never mix with old ones
OUTCOMES_FILE = os.getenv("OUTCOMES_FILE", "df_longitudinal_outcomes.csv")
OUTCOME_LIST_FILE = os.getenv("OUTCOME_LIST_FILE", "")
RESULTS_TAG = os.getenv("RESULTS_TAG", "")

FEATURE_SELECTION = os.getenv("FEATURE_SELECTION", "none")
TOP_K = int(os.getenv("TOP_K", "20"))
RF_TREES = int(os.getenv("RF_TREES", "100"))
RF_JOBS = int(os.getenv("OMP_NUM_THREADS", "4"))

if FEATURE_SELECTION not in ("none", "rf_infold", "rf_fulldata"):
    raise ValueError(f"Unknown FEATURE_SELECTION: {FEATURE_SELECTION}")


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------
# Outcome loading (one or several csv files, merged on "subject")
# ----------------------------------------------------------------------
def read_table(path):
    """Read a csv whose first column is either a row index (as in the original
    files) or already 'subject'."""
    first = pd.read_csv(path, nrows=0).columns[0]
    return pd.read_csv(path, index_col=None if first == "subject" else 0)


def load_outcomes():
    """Returns (df with 'subject' first + one column per outcome, list of outcomes)."""
    wanted = None
    if OUTCOME_LIST_FILE:
        with open(OUTCOME_LIST_FILE) as fh:
            wanted = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
        wanted = list(dict.fromkeys(wanted))  # drop duplicates, keep order

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
        outcomes = merged.columns[1:].tolist()      # original behaviour
    else:
        outcomes = [w for w in wanted if w in merged.columns]
        missing = [w for w in wanted if w not in merged.columns]
        if missing:
            log(f"ERROR: {len(missing)} requested outcome(s) not found: {missing}")
            raise SystemExit(1)
    return merged[["subject"] + outcomes], outcomes


# ----------------------------------------------------------------------
# Random Forest feature ranking
# ----------------------------------------------------------------------
def rf_top_k(X, y, k, seed=4):
    """Indices of the k features with the highest RF impurity importance."""
    rf = RandomForestRegressor(
        n_estimators=RF_TREES, random_state=seed, n_jobs=RF_JOBS
    )
    rf.fit(X, y)
    order = np.argsort(rf.feature_importances_)[::-1]
    return np.sort(order[:k]), rf.feature_importances_


# ----------------------------------------------------------------------
# TabPFN construction: only pass arguments this TabPFN version accepts.
# ----------------------------------------------------------------------
def make_tabpfn(device):
    wanted = {
        "device": device,
        "n_estimators": 1,
        "ignore_pretraining_limits": True,
        "random_state": 4,
    }
    accepted = set(inspect.signature(TabPFNRegressor.__init__).parameters)
    kwargs = {k: v for k, v in wanted.items() if k in accepted}
    dropped = sorted(set(wanted) - set(kwargs))
    log(f"TabPFN version: {tabpfn.__version__}")
    log(f"TabPFN kwargs used: {kwargs}")
    if dropped:
        log(f"TabPFN kwargs NOT supported by this version (skipped): {dropped}")
    return TabPFNRegressor(**kwargs)


def benchmark_tabpfn(model, X, y, n_runs=3):
    """Time a few single (RF selection +) fit + predict calls."""
    n_train = int(len(y) * 0.75)
    for i in range(n_runs):
        Xtr, ytr, Xte = X[:n_train], y[:n_train], X[n_train:]
        t_rf = 0.0
        if FEATURE_SELECTION == "rf_infold":
            t = time.time()
            idx, _ = rf_top_k(Xtr, ytr, TOP_K)
            t_rf = time.time() - t
            Xtr, Xte = Xtr[:, idx], Xte[:, idx]
        t0 = time.time()
        model.fit(Xtr, ytr)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
        model.predict(Xte)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        rf_txt = f"RF select {t_rf:.2f}s  " if FEATURE_SELECTION == "rf_infold" else ""
        log(f"  benchmark run {i}: {rf_txt}fit {t1 - t0:.2f}s  "
            f"predict {time.time() - t1:.2f}s")


def tabpfn_permutation_test(model, X, y, cv, n_perm, seed=4):
    """
    Same computation as sklearn's permutation_test_score (same CV folds,
    y shuffled over the whole dataset, mean negative MSE across folds,
    p = (C + 1) / (n_perm + 1)), reusing one model object.

    With FEATURE_SELECTION == "rf_infold", the RF top-K selection is run
    inside each training fold, for the real AND every permuted outcome.
    Returns the per-fold selected feature indices of the unpermuted run.
    """
    splits = list(cv.split(X))

    def cv_score(yv, record=None):
        scores = []
        for tr, te in splits:
            Xtr, Xte = X[tr], X[te]
            if FEATURE_SELECTION == "rf_infold":
                idx, _ = rf_top_k(Xtr, yv[tr], TOP_K)
                Xtr, Xte = Xtr[:, idx], Xte[:, idx]
                if record is not None:
                    record.append(idx)
            model.fit(Xtr, yv[tr])
            scores.append(-mean_squared_error(yv[te], model.predict(Xte)))
        return float(np.mean(scores))

    t0 = time.time()
    fold_selections = []
    score = cv_score(y, record=fold_selections)
    log(f"Unpermuted CV score: {score:.6g}  ({time.time() - t0:.1f}s)")

    rng = np.random.RandomState(seed)
    perm_scores = np.empty(n_perm)
    for i in range(n_perm):
        perm_scores[i] = cv_score(y[rng.permutation(len(y))])
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 2) * (n_perm - i - 1)
            log(f"  perm {i + 1}/{n_perm}  elapsed {elapsed / 60:.1f} min  ETA {eta / 60:.1f} min")

    pvalue = (np.sum(perm_scores >= score) + 1.0) / (n_perm + 1)
    return score, perm_scores, pvalue, fold_selections


def main():
    log(f"Script version: {SCRIPT_VERSION}")
    log(f"Python: {sys.executable}")
    log(f"Torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"OMP_NUM_THREADS={os.getenv('OMP_NUM_THREADS')}  MODEL_TYPE={MODEL_TYPE}  N_PERM={N_PERM}")
    log(f"FEATURE_SELECTION={FEATURE_SELECTION}  TOP_K={TOP_K}  RF_TREES={RF_TREES}")

    log(f"OUTCOMES_FILE={OUTCOMES_FILE}  OUTCOME_LIST_FILE={OUTCOME_LIST_FILE or '(none: all columns)'}"
        f"  RESULTS_TAG={RESULTS_TAG or '(none)'}")
    df_outcomes, outcomes = load_outcomes()
    df_predictors = sorted(glob.glob(data_dir + "*_400Parcels_bilateral.csv"))
    all_pairs = [(outcome, pred) for outcome in outcomes for pred in df_predictors]
    log(f"Total outcome x predictor pairs: {len(all_pairs)}")

    lsb_index = os.getenv("LSB_JOBINDEX")
    if lsb_index is not None and int(lsb_index) > 0:
        pair_idx = int(lsb_index) - 1
        if pair_idx >= len(all_pairs):
            log(f"Index {lsb_index} is out of range for {len(all_pairs)} pairs. Exiting.")
            return
        outcome_var, df_predictor = all_pairs[pair_idx]
        log(f"--- Running Array Task {lsb_index}/{len(all_pairs)}: {outcome_var} x {df_predictor} ---")
        predict_longitudinal_outcome(df_predictor, df_outcomes, outcome_var, MODEL_TYPE, n_perm=N_PERM)
    else:
        log("No LSB_JOBINDEX found. Running all pairs sequentially...")
        for outcome_var, df_predictor in all_pairs:
            predict_longitudinal_outcome(df_predictor, df_outcomes, outcome_var, MODEL_TYPE, n_perm=N_PERM)


def predict_longitudinal_outcome(df_predictor, df_outcomes, outcome_var, model_type,
                                 multioutput=False, n_perm=100):
    t_start = time.time()
    predictor_name = df_predictor.split(".")[0].split("/")[-1]
    log("\n")
    log(f"Outcome: {outcome_var}")
    log(f"Predictor: {predictor_name}")

    df_pred = pd.read_csv(df_predictor, index_col=0)
    df_model = pd.merge(
        df_outcomes[["subject", outcome_var]], df_pred, on="subject", how="outer"
    ).dropna(how="any")
    log(f"Dataset shape: {df_model.shape}")

    feature_names = np.array(df_model.columns[2:])
    X = df_model.iloc[:, 2:].to_numpy(dtype=np.float32)
    y = df_model[outcome_var].to_numpy(dtype=np.float32)
    cv_strategy = KFold(n_splits=4, shuffle=True, random_state=4)

    use_fs = model_type == "tabpfn" and FEATURE_SELECTION != "none"
    fulldata_idx = None
    fold_selections = []

    # ------------------------------------------------------------------
    # 1. MODEL DEFINITION
    # ------------------------------------------------------------------
    if model_type == "ridge":
        final_model = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-4, 5, 50), scoring="neg_mean_squared_error",
                    alpha_per_target=multioutput),
        )

    elif model_type == "xgboost":
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("xgb", XGBRegressor(objective="reg:squarederror", random_state=4,
                                 n_jobs=1, tree_method="hist")),
        ])
        param_grid = {
            "xgb__n_estimators": [50, 100, 200],
            "xgb__max_depth": [2, 3, 5],
            "xgb__learning_rate": [0.01, 0.05, 0.1],
            "xgb__subsample": [0.8, 1.0],
            "xgb__colsample_bytree": [0.5, 0.8, 1.0],
        }
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv_strategy,
                                   scoring="neg_mean_squared_error", n_jobs=4, verbose=1)
        grid_search.fit(X, y)
        final_model = grid_search.best_estimator_

    elif model_type == "svr":
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svr", LinearSVR(dual="auto", tol=1e-3)),
        ])
        param_grid = {
            "svr__C": [0.01, 0.1, 1.0, 10],
            "svr__epsilon": [0.001, 0.01, 0.1, 0.5, 0.75],
        }
        grid_search = HalvingGridSearchCV(pipeline, param_grid, cv=cv_strategy,
                                          scoring="neg_mean_squared_error")
        grid_search.fit(X, y)
        final_model = grid_search.best_estimator_

    elif model_type == "tabpfn":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"--- DETECTED DEVICE FOR TABPFN: {device} ---")
        final_model = make_tabpfn(device)

        if FEATURE_SELECTION == "rf_fulldata":
            # Selection ONCE on all subjects (biased; see note at top of file)
            fulldata_idx, _ = rf_top_k(X, y, TOP_K)
            X = X[:, fulldata_idx]
            log(f"RF top-{TOP_K} on FULL data (fixed for CV + permutations): "
                f"{list(feature_names[fulldata_idx])}")

        log("Timing single fit/predict (first run includes model loading):")
        benchmark_tabpfn(final_model, X, y)
        log("Expected permutation time ~ (n_perm+1) x 4 x (RF select + fit + predict) above")

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # ------------------------------------------------------------------
    # 2. PERMUTATION TESTING
    # ------------------------------------------------------------------
    log(f"Starting permutation test (n_perm={n_perm})...")
    if model_type == "tabpfn":
        score, permutation_scores, pvalue, fold_selections = tabpfn_permutation_test(
            final_model, X, y, cv_strategy, n_perm, seed=4
        )
    else:
        score, permutation_scores, pvalue = permutation_test_score(
            estimator=final_model, X=X, y=y, scoring="neg_mean_squared_error",
            cv=cv_strategy, n_permutations=n_perm, n_jobs=4, verbose=1, random_state=4,
        )

    log(f"Original score: {score:.6g}")
    log(f"Permutation p-value: {pvalue:.4f}")

    if model_type == "xgboost":
        output_dir = "xgBoost"
    elif use_fs:
        output_dir = f"tabpfn_{FEATURE_SELECTION}_top{TOP_K}"
    else:
        output_dir = model_type.lower()
    if RESULTS_TAG:
        output_dir = f"{output_dir}_{RESULTS_TAG}"
    os.makedirs(output_dir, exist_ok=True)
    tag = f"baseline_{predictor_name}_{model_type}_{outcome_var}"

    g = sns.displot(permutation_scores, kind="hist", kde=True)
    g.refline(x=score, color="red", linewidth=2)
    g.fig.suptitle(f"{predictor_name} predicting {outcome_var}", y=1.02)
    plot_filename = f"{output_dir}/{tag}.png"
    g.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close("all")

    results_filename = f"{output_dir}/{tag}_results.txt"
    with open(results_filename, "w") as f:
        f.write(f"Predictor: {predictor_name}\n")
        f.write(f"Outcome: {outcome_var}\n")
        f.write(f"Model: {model_type}\n")
        f.write(f"Dataset Shape: {df_model.shape}\n")
        f.write(f"Feature selection: {FEATURE_SELECTION if use_fs else 'none'}\n")
        f.write(f"Number of features used: {TOP_K if use_fs else X.shape[1]}\n")
        f.write(f"Number of permutations: {n_perm}\n")
        f.write(f"Original MSE Score: {score:.6g}\n")
        f.write(f"Permutation p-value: {pvalue:.6f}\n")
        f.write(f"Runtime (min): {(time.time() - t_start) / 60:.2f}\n")
    np.save(f"{output_dir}/{tag}_permscores.npy", permutation_scores)

    # ------------------------------------------------------------------
    # Which features were selected
    # ------------------------------------------------------------------
    if use_fs:
        if FEATURE_SELECTION == "rf_infold":
            # How many of the 4 training folds picked each feature
            counts = np.zeros(len(feature_names), dtype=int)
            for idx in fold_selections:
                counts[idx] += 1
            # RF on all subjects, for reporting only (not used for the score)
            _, imp_full = rf_top_k(X, y, TOP_K)
            sel = pd.DataFrame({
                "feature": feature_names,
                "folds_selected_of_4": counts,
                "rf_importance_full_data": imp_full,
            })
            sel = sel[(sel.folds_selected_of_4 > 0)].sort_values(
                ["folds_selected_of_4", "rf_importance_full_data"], ascending=False)
            log(f"Features selected in all 4 folds: {(sel.folds_selected_of_4 == 4).sum()} of {TOP_K}")
        else:
            sel = pd.DataFrame({"feature": feature_names[fulldata_idx]})
        sel_filename = f"{output_dir}/{tag}_selected_features.csv"
        sel.to_csv(sel_filename, index=False)
        log(f"Saved selected features to: {sel_filename}")

    log(f"Saved plot to: {plot_filename}")
    log(f"Saved metrics to: {results_filename}")

    # ------------------------------------------------------------------
    # 3. POST-PROCESSING & FEATURE IMPORTANCE
    # ------------------------------------------------------------------
    if model_type == "ridge":
        final_model.fit(X, y)
        log(f"Best Parameters (Alpha): {final_model.named_steps['ridgecv'].alpha_}")
    elif model_type == "xgboost":
        final_model.fit(X, y)
        importance = pd.Series(
            final_model.named_steps["xgb"].feature_importances_, index=df_model.columns[2:]
        ).sort_values(ascending=False)
        log(f"\nTop 20 Features:\n{importance.head(20)}")
    elif model_type == "svr":
        final_model.fit(X, y)
        log("SVR Model fitted successfully.")

    log(f"Total time for this pair: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
