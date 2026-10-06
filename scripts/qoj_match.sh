#!/usr/bin/env bash
set -euo pipefail
DDZ_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DDZ_MATCH_ROOT="${DDZ_MATCH_ROOT:-$DDZ_REPO_ROOT/runs/qoj_match_v1}"
DDZ_SESSION="${DDZ_SESSION:-ddz_qoj_match}"
DDZ_PYTHON="${DDZ_PYTHON:-$DDZ_REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$DDZ_PYTHON" ]]; then DDZ_PYTHON="$DDZ_REPO_ROOT/../.venv-gnn-jax/bin/python"; fi
if [[ ! -x "$DDZ_PYTHON" ]]; then DDZ_PYTHON="$(command -v python3)"; fi
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$DDZ_REPO_ROOT"
case "${1:-status}" in
  prepare)
    : "${DDZ_USERNAME:?Set DDZ_USERNAME to your authenticated account name}"
    exec "$DDZ_PYTHON" -m ddz.qoj_deployment prepare --repo "$DDZ_REPO_ROOT" \
      --root "$DDZ_MATCH_ROOT" --models "${DDZ_MODELS:-$DDZ_REPO_ROOT/models}" \
      --token-file "${DDZ_TOKEN_FILE:-$HOME/.config/ddz/qoj_api_key}" --username "$DDZ_USERNAME" \
      --base "${DDZ_BASE:-https://qoj.ac/api/v1/doudizhu}" --session "$DDZ_SESSION" \
      --checkpoint-pointer "${DDZ_CHECKPOINT_POINTER:-$DDZ_REPO_ROOT/runs/production_current.json}"
    ;;
  start)
    if [[ -f "$DDZ_MATCH_ROOT/OFFLINE.json" && "${DDZ_QOJ_ENABLE_ONLINE:-0}" != 1 ]]; then
      echo 'QOJ is disabled by the user; do not restart until the user requests online play.'
      exit 77
    fi
    if tmux has-session -t "$DDZ_SESSION" 2>/dev/null; then
      echo "Session $DDZ_SESSION already exists; attach or inspect it before restarting."
      exit 1
    fi
    [[ -x "$DDZ_MATCH_ROOT/launch_current.sh" ]] || { echo 'Run prepare first'; exit 1; }
    printf -v DDZ_START_CMD '%q' "$DDZ_MATCH_ROOT/launch_current.sh"
    tmux new-session -d -s "$DDZ_SESSION" -x 150 -y 45 "$DDZ_START_CMD"
    tmux set-window-option -t "$DDZ_SESSION:0" remain-on-exit on
    printf -v DDZ_LOG_CMD 'tail -F %q' "$DDZ_MATCH_ROOT/console.log"
    tmux split-window -v -l 4 -t "$DDZ_SESSION:0" "$DDZ_LOG_CMD"
    # A 24-row attached terminal still needs room for the full state/value view.
    # Scope both hooks to this session; other tmux sessions are unaffected.
    tmux set-hook -t "$DDZ_SESSION" client-attached "resize-pane -t $DDZ_SESSION:0.1 -y 4"
    tmux set-hook -t "$DDZ_SESSION" client-resized "resize-pane -t $DDZ_SESSION:0.1 -y 4"
    tmux select-pane -t "$DDZ_SESSION:0.0"
    ;;
  attach) exec tmux attach-session -t "$DDZ_SESSION" ;;
  status) exec "$DDZ_PYTHON" -c 'import json,sys; from pathlib import Path; print(json.dumps(json.loads((Path(sys.argv[1])/"status.json").read_text()),ensure_ascii=False,indent=2))' "$DDZ_MATCH_ROOT" ;;
  *) echo 'Usage: qoj_match.sh {prepare|start|attach|status}'; exit 2 ;;
esac
