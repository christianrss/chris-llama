#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TARGETS="${ACPP_TARGETS:-generic}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)}"

if [[ -x "$PROJECT_ROOT/.deps/adaptivecpp/bin/acpp" ]]; then
    ACPP_BIN="$PROJECT_ROOT/.deps/adaptivecpp/bin/acpp"
elif command -v acpp >/dev/null 2>&1; then
    ACPP_BIN="$(command -v acpp)"
else
    echo "error: AdaptiveCpp is not installed" >&2
    echo "Run: ./scripts/setup_adaptivecpp.sh" >&2
    exit 1
fi

"$ACPP_BIN" --version
make -C "$PROJECT_ROOT" clean
make -C "$PROJECT_ROOT" BACKEND=acpp ACPP="$ACPP_BIN" ACPP_TARGETS="$TARGETS" -j"$JOBS"

echo
echo "Build complete. Visible devices:"
"$PROJECT_ROOT/build/bin/chris_llama" --list-devices
