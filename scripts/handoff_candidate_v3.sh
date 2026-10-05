#!/usr/bin/env bash
set -euo pipefail
old_pid="${1:?old trainer pid required}"
while kill -0 "$old_pid" 2>/dev/null; do
  sleep 5
done
exec /mnt/pfs/guoyuchong/guoyuchong/ddz/scripts/resume_candidate_v3.sh
