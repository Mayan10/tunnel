#!/usr/bin/env sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

if command -v pipx >/dev/null 2>&1; then
  pipx install --force "${ROOT_DIR}"
else
  python3 -m pip install --user --upgrade "${ROOT_DIR}"
  printf 'Installed with pip. For an isolated install instead, use pipx: https://pipx.pypa.io\n'
fi

printf 'Installed tunnel.\n'
printf 'If the command is not found, add this to your shell config:\n'
printf '  export PATH="$HOME/.local/bin:$PATH"\n'
