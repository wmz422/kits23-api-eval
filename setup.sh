#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" -gt 1 || ( "$#" -eq 1 && "$1" != "--viewer" && "$1" != "--help" ) ]]; then
  echo 'Usage: bash setup.sh [--viewer]' >&2
  exit 2
fi
if [[ "${1:-}" == "--help" ]]; then
  echo 'Usage: bash setup.sh [--viewer] (include optional saved-results viewer dependency)'
  exit 0
fi
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
kits23_python="${KITS23_PYTHON:-python3}"
"$kits23_python" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else "Python 3.10+ is required")'
if [[ ! -x "$repo_dir/.venv/bin/python" ]]; then
  "$kits23_python" -m venv "$repo_dir/.venv" || {
    echo 'Cannot create venv. Install Python 3.10+ with venv support (Ubuntu/Debian: python3-venv), or set KITS23_PYTHON.' >&2
    exit 2
  }
fi
"$repo_dir/.venv/bin/python" -m pip install --disable-pip-version-check -r "$repo_dir/requirements.txt"
if [[ "${1:-}" == "--viewer" ]]; then
  "$repo_dir/.venv/bin/python" -m pip install --disable-pip-version-check -r "$repo_dir/requirements-viewer.txt"
fi
"$repo_dir/.venv/bin/python" -m pip freeze > "$repo_dir/.venv/installed-requirements.txt"
echo "Ready. Run: $repo_dir/.venv/bin/python $repo_dir/run.py --base-url http://127.0.0.1:8000/v1 --model YOUR_MODEL_ID"
