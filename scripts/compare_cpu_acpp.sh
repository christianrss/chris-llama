#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if [[ $# -lt 2 ]]; then
    echo "usage: $0 MODEL.gguf 'prompt'" >&2
    exit 2
fi

MODEL="$(realpath "$1")"
PROMPT="$2"
TARGETS="${ACPP_TARGETS:-generic}"

if [[ -x "$PROJECT_ROOT/.deps/adaptivecpp/bin/acpp" ]]; then
    ACPP_BIN="$PROJECT_ROOT/.deps/adaptivecpp/bin/acpp"
elif command -v acpp >/dev/null 2>&1; then
    ACPP_BIN="$(command -v acpp)"
else
    echo "error: AdaptiveCpp was not found; run ./scripts/setup_adaptivecpp.sh" >&2
    exit 1
fi

make -C "$PROJECT_ROOT" clean >/dev/null
make -C "$PROJECT_ROOT" BACKEND=cpu -j >/dev/null
CPU_ID="$("$PROJECT_ROOT/build/bin/chris_llama" "$MODEL" --device cpu -p "$PROMPT" --next-token-id --temperature 0)"

make -C "$PROJECT_ROOT" clean >/dev/null
make -C "$PROJECT_ROOT" BACKEND=acpp ACPP="$ACPP_BIN" ACPP_TARGETS="$TARGETS" -j >/dev/null
ACPP_ID="$("$PROJECT_ROOT/build/bin/chris_llama" "$MODEL" --device gpu -p "$PROMPT" --next-token-id --temperature 0)"

printf 'CPU next token : %s\n' "$CPU_ID"
printf 'ACPP next token: %s\n' "$ACPP_ID"

if [[ "$CPU_ID" != "$ACPP_ID" ]]; then
    echo "MISMATCH" >&2
    exit 1
fi

echo "OK: backend results match."
