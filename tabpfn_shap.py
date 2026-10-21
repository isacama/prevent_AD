"""
SHAP feature importance for TabPFN (prevent-AD).

For each requested outcome x brain-measure pair:
  * 4-fold CV with the SAME folds as the main analysis (KFold, shuffle, random_state=4)
  * in each fold: fit TabPFN on the training people, then compute SHAP values
    for the held-out people (out-of-sample explanations)
  * combine the folds and save per-region and per-network summaries + plots

SHAP is descriptive: it explains what the fitted model relies on. It is not a
significance test. Only interpret it for pairs where the model predicts better
than chance (permutation p < .05).

Settings come from environment variables (set them in the job file):
  SHAP_PAIRS      comma-separated "outcome:predictor" list, e.g.
                  "pTau_intercept:rsfc_perirhinal,AB_ratio_slope:rsfc_hipp_body"
                  LSB_JOBINDEX picks one entry (1-based). Without LSB_JOBINDEX
                  all entries run one after another.
  SHAP_FEATURES   "all" (default, all 200 regions) or "rf_fulldata_top20"
                  (the 20 regions saved by the rf_fulldata run)
  SHAP_BACKGROUND training people used as the "average person" baseline (default 50)
  SHAP_EXPLAIN    max held-out people explained per fold (default 0 = all)
  SHAP_PASSES     permutation passes per person; more = less noisy, slower (default 1)
  SHAP_BATCH      SHAP evaluations per TabPFN predict call (default 50)

Run once without a job to list valid outcome/predictor names:
  python tabpfn_shap.py --list
"""
import os

PROJECT_DIR = "/projectnb/jamlab/ixc751"
os.environ.setdefault("HOME", PROJECT_DIR)
os.environ.setdefault("XDG_CACHE_HOME", f"{PROJECT_DIR}/.cache")
os.environ.setdefault("TABPFN_CACHE_DIR", f"{PROJECT_DIR}/.cache/tabpfn")
os.environ.setdefault("HF_HOME", f"{PROJECT_DIR}/.cache/huggingface")
os.environ.setdefault("TORCH_HOME", f"{PROJECT_DIR}/.cache/torch")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import glob
import inspect
import re
import sys
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.model_selection import KFold
from tabpfn import TabPFNRegressor

warnings.filterwarnings("ignore")

SCRIPT_VERSION = "2026-10-13 tabpfn_shap v1"
data_dir = "/projectnb/jamlab/prevent_AD_dataset/prevent_AD_dataset/"

SHAP_FEATURES = os.getenv("SHAP_FEATURES", "all")
N_BACKGROUND = int(os.getenv("SHAP_BACKGROUND", "50"))
N_EXPLAIN = int(os.getenv("SHAP_EXPLAIN", "0"))
N_PASSES = int(os.getenv("SHAP_PASSES", "1"))
BATCH = int(os.getenv("SHAP_BATCH", "50"))
SEED = 4


def log(msg):
    print(msg, flush=True)


def make_tabpfn(device):
    wanted = {"device": device, "n_estimators": 1,
              "ignore_pretraining_limits": True, "random_state": SEED}
    accepted = set(inspect.signature(TabPFNRegressor.__init__).parameters)
    return TabPFNRegressor(**{k: v for k, v in wanted.items() if k in accepted})


def network_of(feature):
    """'17networks_LH_DefaultC_PHC_1' -> 'DefaultC' (falls back to the full name)."""
    m = re.match(r"\d+networks_[LR]H_([^_]+)", str(feature))
    return m.group(1) if m else str(feature)


def find_predictor_file(name):
    files = sorted(glob.glob(data_dir + "*_400Parcels_bilateral.csv"))
    for f in files:
        short = os.path.basename(f).replace("_400Parcels_bilateral.csv", "")
        if name in (short, os.path.basename(f), os.path.basename(f).split(".")[0]):
            return f
    raise ValueError(f"Predictor '{name}' not found. Options: "
                     f"{[os.path.basename(f).replace('_400Parcels_bilateral.csv', '') for f in files]}")


def run_pair(outcome, predictor, device):
    t_start = time.time()
    pred_file = find_predictor_file(predictor)
    pred_name = os.path.basename(pred_file).split(".")[0]
    log(f"\n=== SHAP: {outcome} x {pred_name} ===")

    df_out = pd.read_csv(data_dir + "df_longitudinal_outcomes.csv", index_col=0)
    df_pred = pd.read_csv(pred_file, index_col=0)
    df = pd.merge(df_out[["subject", outcome]], df_pred, on="subject", how="outer").dropna(how="any")
    feature_names = np.array(df.columns[2:])

    if SHAP_FEATURES == "rf_fulldata_top20":
        sel_file = (f"tabpfn_rf_fulldata_top20/baseline_{pred_name}_tabpfn_{outcome}"
                    f"_selected_features.csv")
        keep = pd.read_csv(sel_file)["feature"].tolist()
        feature_names = np.array([f for f in feature_names if f in keep])
        log(f"Using {len(feature_names)} features from {sel_file}")
    elif SHAP_FEATURES != "all":
        raise ValueError(f"Unknown SHAP_FEATURES: {SHAP_FEATURES}")

    X = df[feature_names].to_numpy(dtype=np.float32)
    y = df[outcome].to_numpy(dtype=np.float32)
    M = X.shape[1]
    max_evals = N_PASSES * (2 * M + 1)
    log(f"Subjects: {len(y)}  features: {M}  background: {N_BACKGROUND}  "
        f"evals/person: {max_evals}  (~{max_evals * N_BACKGROUND:,} TabPFN rows predicted per person)")

    cv = KFold(n_splits=4, shuffle=True, random_state=SEED)
    rng = np.random.RandomState(SEED)
    model = make_tabpfn(device)

    all_shap, all_X, fold_rank = [], [], []
    for k, (tr, te) in enumerate(cv.split(X), start=1):
        t0 = time.time()
        model.fit(X[tr], y[tr])
        bg = X[tr][rng.choice(len(tr), size=min(N_BACKGROUND, len(tr)), replace=False)]
        te_use = te if N_EXPLAIN <= 0 else te[rng.choice(len(te), size=min(N_EXPLAIN, len(te)),
                                                          replace=False)]
        explainer = shap.Explainer(lambda Z: model.predict(np.asarray(Z, dtype=np.float32)),
                                   shap.maskers.Independent(bg), algorithm="permutation",
                                   feature_names=list(feature_names))
        ex = explainer(X[te_use], max_evals=max_evals, batch_size=BATCH, silent=True)
        all_shap.append(ex.values)
        all_X.append(X[te_use])
        fold_imp = np.abs(ex.values).mean(axis=0)
        fold_rank.append(pd.Series(fold_imp, index=feature_names).rank(ascending=False))
        log(f"  fold {k}: explained {len(te_use)} people in {(time.time() - t0) / 60:.1f} min")

    S = np.vstack(all_shap)
    XV = np.vstack(all_X)

    # ---------------- per-region summary ----------------
    direction = []
    for j in range(M):
        if np.std(XV[:, j]) > 0 and np.std(S[:, j]) > 0:
            direction.append(np.corrcoef(XV[:, j], S[:, j])[0, 1])
        else:
            direction.append(np.nan)
    ranks = pd.concat(fold_rank, axis=1)
    top_n = min(20, M)
    region = pd.DataFrame({
        "feature": feature_names,
        "network": [network_of(f) for f in feature_names],
        "mean_abs_shap": np.abs(S).mean(axis=0),        # how much the model relies on it
        "mean_shap": S.mean(axis=0),                    # ~0 by construction; sanity check
        "direction_r": direction,                       # + : higher value -> higher prediction
        "folds_in_top20_of_4": (ranks <= top_n).sum(axis=1).values,
    }).sort_values("mean_abs_shap", ascending=False)
    region["rank"] = np.arange(1, M + 1)

    # ---------------- per-network summary ----------------
    abs_df = pd.DataFrame(np.abs(S), columns=feature_names)
    net_tot = abs_df.T.groupby([network_of(f) for f in feature_names]).sum().T  # per person
    network = pd.DataFrame({
        "network": net_tot.columns,
        "sum_mean_abs_shap": net_tot.mean(axis=0).values,
        "n_regions": [sum(network_of(f) == n for f in feature_names) for n in net_tot.columns],
    })
    network["mean_abs_shap_per_region"] = network.sum_mean_abs_shap / network.n_regions
    network["share_of_total_pct"] = 100 * network.sum_mean_abs_shap / network.sum_mean_abs_shap.sum()
    network = network.sort_values("sum_mean_abs_shap", ascending=False)

    out_dir = f"shap_tabpfn_{SHAP_FEATURES}"
    os.makedirs(out_dir, exist_ok=True)
    tag = f"shap_{pred_name}_{outcome}"
    region.to_csv(f"{out_dir}/{tag}_regions.csv", index=False)
    network.to_csv(f"{out_dir}/{tag}_networks.csv", index=False)
    np.savez_compressed(f"{out_dir}/{tag}_values.npz", shap=S, X=XV, features=feature_names)

    # ---------------- plots ----------------
    try:
        expl = shap.Explanation(values=S, data=XV, feature_names=list(feature_names))
        plt.figure()
        shap.plots.beeswarm(expl, max_display=20, show=False)
        plt.title(f"{pred_name} -> {outcome}", fontsize=10)
        plt.savefig(f"{out_dir}/{tag}_beeswarm.png", dpi=200, bbox_inches="tight")
        plt.close("all")

        fig, ax = plt.subplots(figsize=(6, 0.35 * len(network) + 1.5))
        nd = network.sort_values("sum_mean_abs_shap")
        ax.barh(nd.network, nd.sum_mean_abs_shap, color="#4C72B0")
        ax.set_xlabel("Total mean |SHAP| (summed over regions in network)")
        ax.set_title(f"{pred_name} -> {outcome}: importance by network", fontsize=10)
        plt.savefig(f"{out_dir}/{tag}_networks.png", dpi=200, bbox_inches="tight")
        plt.close("all")
    except Exception as e:
        log(f"(plots skipped: {e})")

    log("Top 10 regions by mean |SHAP|:")
    log(region.head(10)[["rank", "feature", "network", "mean_abs_shap", "direction_r",
                         "folds_in_top20_of_4"]].to_string(index=False))
    log("Top networks:")
    log(network.head(8).to_string(index=False))
    log(f"Saved to {out_dir}/{tag}_*  ({(time.time() - t_start) / 60:.1f} min)")


def main():
    log(f"Script version: {SCRIPT_VERSION}  | shap {shap.__version__} | torch {torch.__version__}")
    if "--list" in sys.argv:
        df_out = pd.read_csv(data_dir + "df_longitudinal_outcomes.csv", index_col=0)
        log(f"Outcomes: {df_out.columns[1:].tolist()}")
        log("Predictors: " + str([os.path.basename(f).replace("_400Parcels_bilateral.csv", "")
                                   for f in sorted(glob.glob(data_dir + '*_400Parcels_bilateral.csv'))]))
        return

    pairs = [p.strip().split(":") for p in os.getenv("SHAP_PAIRS", "").split(",") if p.strip()]
    if not pairs:
        sys.exit("Set SHAP_PAIRS, e.g. SHAP_PAIRS='pTau_intercept:rsfc_perirhinal'")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}  SHAP_FEATURES={SHAP_FEATURES}")

    idx = os.getenv("LSB_JOBINDEX")
    if idx is not None and int(idx) > 0:
        if int(idx) > len(pairs):
            log(f"Task {idx} > {len(pairs)} pairs in SHAP_PAIRS. Exiting.")
            return
        pairs = [pairs[int(idx) - 1]]
    for outcome, predictor in pairs:
        run_pair(outcome, predictor, device)


if __name__ == "__main__":
    main()
