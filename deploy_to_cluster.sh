#!/bin/bash
# Copy the current ("master") code from the JupyterHub project folder to the cluster working folder.
# Run in the Pegasus terminal:   bash /nethome/ixc751/prevent_AD/deploy_to_cluster.sh
#
# What it does
#   * copies the active scripts, job files and the outcome list to the cluster working folder (flat)
#   * removes Windows line endings
#   * puts your TabPFN token into the job files from a private file (so job files in JupyterHub can
#     keep the placeholder PASTE_TOKEN_HERE). One-time setup, run in the cluster working folder:
#         grep '^export TABPFN_TOKEN' prevent-AD_tabpfn_rf_cpu.job > .tabpfn_env ; chmod 600 .tabpfn_env
#   * prints script versions, and lists files in the cluster folder that this script does not manage
#     (candidates to move into _archive/ - it never deletes anything)
SRC="${SRC:-/nethome/ixc751/prevent_AD}"
DST="${DST:-/projectnb/jamlab/ixc751}"

# warn if jobs are queued/running: overwriting a script just as a task starts could break that task
if command -v bjobs >/dev/null 2>&1 && [ "${FORCE:-0}" != "1" ]; then
  n=$(bjobs 2>/dev/null | grep -vc "JOBID\|jupyter")
  if [ "$n" -gt 0 ]; then
    echo "WARNING: $n job(s) queued or running (bjobs). Copying identical files is usually harmless,"
    echo "but do not move/rename/delete files in $DST now."
    read -p "Continue? [y/N] " a; [ "$a" = "y" ] || exit 1
  fi
fi

DEPLOYED=()
copy() {   # copy <source path>
  if [ -f "$1" ]; then
    cp "$1" "$DST/" && DEPLOYED+=("$(basename "$1")")
  else
    echo "  (missing, skipped: $1)"
  fi
}

echo "Copying from $SRC to $DST"
# --- scripts
copy "$SRC/tabpfn/prevent-AD_sklearn_tabpfn_flexible.py"
copy "$SRC/regressions/prevent-AD_sklearn_regressions_flexible.py"
copy "$SRC/tabpfn/shap/tabpfn_shap.py"
copy "$SRC/tabpfn/shap/summarize_tabpfn_shap.py"
copy "$SRC/shared/summarize_tabpfn_results.py"
copy "$SRC/shared/pairs_to_tasks.py"
copy "$SRC/shared/outcomes_new_blup.txt"
# --- job files (every .job in the active jobs folders)
for f in "$SRC"/tabpfn/jobs/*.job "$SRC"/regressions/jobs/*.job "$SRC"/tabpfn/shap/*.job; do
  [ -f "$f" ] && copy "$f"
done

# --- line endings
for b in "${DEPLOYED[@]}"; do sed -i 's/\r$//' "$DST/$b"; done

# --- token
if [ -f "$DST/.tabpfn_env" ]; then
  tok=$(cat "$DST/.tabpfn_env")
  for b in "${DEPLOYED[@]}"; do
    case "$b" in *.job)
      awk -v t="$tok" '/^export TABPFN_TOKEN=/{print t; next}{print}' "$DST/$b" > "$DST/$b.tmp" && mv "$DST/$b.tmp" "$DST/$b";;
    esac
  done
  echo "Token inserted into job files from $DST/.tabpfn_env"
else
  echo "NOTE: $DST/.tabpfn_env not found; job files still contain whatever token line they had."
fi
left=$(grep -l "PASTE_TOKEN_HERE" "$DST"/*.job 2>/dev/null)
[ -n "$left" ] && echo "WARNING: placeholder token still in: $left"

# --- report
echo; echo "Deployed ${#DEPLOYED[@]} files:"; printf '  %s\n' "${DEPLOYED[@]}"
echo; echo "Versions now in $DST:"
grep -H "^SCRIPT_VERSION" "$DST"/prevent-AD_sklearn_tabpfn_flexible.py "$DST"/prevent-AD_sklearn_regressions_flexible.py 2>/dev/null

echo; echo "Files in $DST NOT managed by this script (candidates for _archive/, review before moving):"
ls -p "$DST" | grep -v / | while read -r f; do
  keep=0; for b in "${DEPLOYED[@]}"; do [ "$b" = "$f" ] && keep=1; done
  [ $keep -eq 0 ] && echo "  $f"
done
echo; echo "To archive one: mkdir -p $DST/_archive && mv -n $DST/<file> $DST/_archive/"
