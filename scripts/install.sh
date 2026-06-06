#!/usr/bin/env sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
INSTALL_DIR="${HOME}/.local/bin"
TARGET="${INSTALL_DIR}/tunnel"

python3 "${ROOT_DIR}/scripts/build_zipapp.py"

mkdir -p "${INSTALL_DIR}"
cp "${ROOT_DIR}/dist/tunnel.pyz" "${TARGET}"
chmod +x "${TARGET}"

printf 'Installed tunnel to %s\n' "${TARGET}"
printf 'If needed, add this to your shell config:\n'
printf '  export PATH="$HOME/.local/bin:$PATH"\n'
