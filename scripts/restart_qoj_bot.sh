#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
bash scripts/qoj_match.sh start
exec bash scripts/qoj_match.sh attach
