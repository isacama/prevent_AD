"""
Turn a summary table into LSF array task numbers (for re-running chosen pairs with more permutations).

    python pairs_to_tasks.py tabpfn_newblup/summary_tabpfn_results.csv 0.05
        -> prints task numbers of pairs with p_perm < 0.05, ready for:
           bsub -J "tabpfn_2500perm[<numbers>]" < prevent-AD_tabpfn_2500perm_new_outcomes_cpu.job

Task number = (outcome position in outcomes_new_blup.txt) * 6 + (predictor position), 1-based,
predictors in sorted file-name order: T1T2, rsfc_entorhinal, rsfc_hipp_body, rsfc_hipp_head,
rsfc_hipp_tail, rsfc_perirhinal.
"""
import sys
import pandas as pd

csv = sys.argv[1]
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
col = sys.argv[3] if len(sys.argv) > 3 else "p_perm"
outs = [l.strip() for l in open("outcomes_new_blup.txt") if l.strip() and not l.startswith("#")]
preds = ["T1T2", "rsfc_entorhinal", "rsfc_hipp_body", "rsfc_hipp_head", "rsfc_hipp_tail",
         "rsfc_perirhinal"]
d = pd.read_csv(csv)
d = d[d[col] < thr].copy()
d["task"] = [outs.index(o) * len(preds) + preds.index(p) + 1
             for o, p in zip(d["outcome"], d["predictor"])]
d = d.sort_values("task")
print(d[["task", "outcome", "predictor", col]].to_string(index=False))
print(f"\n{len(d)} pair(s). Array list:\n{','.join(map(str, d['task']))}")
