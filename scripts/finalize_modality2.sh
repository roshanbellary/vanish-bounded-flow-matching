#!/usr/bin/env bash
# Pull the finished Modality 2 rows and rebuild everything that depends on them.
#
# Deliberately does NOT terminate the pod: termination is irreversible and is done by hand
# after the row count has been eyeballed. This script is safe to re-run.
#
#   scripts/finalize_modality2.sh            # pull + rebuild
#   scripts/finalize_modality2.sh --check    # report the remote row count and stop
set -euo pipefail

# No default host: export DGM_IP/DGM_PORT for the current pod before running.
HOST_IP="${DGM_IP:?set DGM_IP to the current pod's IP}"
PORT="${DGM_PORT:?set DGM_PORT to the current pod's SSH port}"
KEY="${DGM_KEY:-$HOME/.ssh/save_key}"
REMOTE="/workspace/dgm/results"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH=(ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
EXPECTED=40

remote_rows() {
  # The remote glob must stay quoted: zsh expands an unquoted one locally and the pull
  # silently matches nothing.
  "${SSH[@]}" "root@$HOST_IP" \
    "cat $REMOTE/modality2_unet_*.jsonl 2>/dev/null | grep -c arm" 2>/dev/null || echo 0
}

n="$(remote_rows)"
echo "remote rows: $n / $EXPECTED"
[ "${1:-}" = "--check" ] && exit 0
if [ "$n" -lt "$EXPECTED" ]; then
  echo "refusing to finalize: $((EXPECTED - n)) cells still missing." >&2
  echo "re-run when the count reaches $EXPECTED, or pass --check to just look." >&2
  exit 1
fi

rsync -az -e "${SSH[*]}" \
  "root@$HOST_IP:$REMOTE/modality2_unet_*.jsonl" \
  "root@$HOST_IP:$REMOTE/samples_*.npy" \
  "$REPO/results/"
echo "pulled $(cat "$REPO"/results/modality2_unet_*.jsonl | grep -c arm) rows"

cd "$REPO"
python3 scripts/make_tables.py
python3 scripts/make_figures.py
( cd paper  && pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 \
             && bibtex main >/dev/null 2>&1 \
             && pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 \
             && pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 )
( cd slides && pdflatex -interaction=nonstopmode defense.tex >/dev/null 2>&1 \
             && pdflatex -interaction=nonstopmode defense.tex >/dev/null 2>&1 )

echo
echo "paper : $(pdfinfo paper/main.pdf | awk '/Pages/{print $2}') pages, \
$(grep -ci 'undefined citation' paper/main.log || true) undefined citations"
echo "slides: $(pdfinfo slides/defense.pdf | awk '/Pages/{print $2}') pages"
echo
echo "Pod is STILL BILLING. Terminate it once these numbers look right."
