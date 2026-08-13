#!/usr/bin/env bash
# Source this file to use the project-local AdaptiveCpp installation.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ACPP_PREFIX="${ACPP_PREFIX:-$PROJECT_ROOT/.deps/adaptivecpp}"

export PATH="$ACPP_PREFIX/bin:$PATH"
export CMAKE_PREFIX_PATH="$ACPP_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

if [[ -d "$ACPP_PREFIX/lib" ]]; then
    export LD_LIBRARY_PATH="$ACPP_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ -d "$ACPP_PREFIX/lib64" ]]; then
    export LD_LIBRARY_PATH="$ACPP_PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
