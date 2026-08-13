#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a project-local AdaptiveCpp installation from Christian's fork.
#
# The script installs only build dependencies. GPU drivers, CUDA and ROCm are
# deliberately left to the host administrator because replacing those stacks
# automatically can break an otherwise working machine.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

ACPP_REPO="${ACPP_REPO:-https://github.com/christianrss/AdaptiveCpp.git}"
ACPP_BRANCH="${ACPP_BRANCH:-develop}"
TESTS_ACPP_REPO="${TESTS_ACPP_REPO:-https://github.com/christianrss/tests-acpp.git}"
DEPS_ROOT="${DEPS_ROOT:-$PROJECT_ROOT/.deps}"
ACPP_SRC="${ACPP_SRC:-$DEPS_ROOT/AdaptiveCpp}"
ACPP_BUILD="${ACPP_BUILD:-$DEPS_ROOT/AdaptiveCpp-build}"
ACPP_PREFIX="${ACPP_PREFIX:-$DEPS_ROOT/adaptivecpp}"
TESTS_ACPP_SRC="${TESTS_ACPP_SRC:-$DEPS_ROOT/tests-acpp}"
ACPP_TARGETS="${ACPP_TARGETS:-generic}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)}"

INSTALL_DEPS=1
CLONE_TESTS_ACPP=0
BUILD_CHRIS_LLAMA=1
RUN_SMOKE_TEST=1
UPDATE_SOURCE=1

usage() {
    cat <<USAGE
usage: $0 [options]

Options:
  --no-deps              Do not install OS build dependencies.
  --with-tests-acpp      Clone/update christianrss/tests-acpp as well.
  --no-build             Install AdaptiveCpp but do not build Chris Llama.
  --no-smoke-test        Skip the minimal SYCL compile/run check.
  --no-update            Do not fetch/pull an existing AdaptiveCpp clone.
  --branch NAME          AdaptiveCpp branch to use (default: develop).
  --targets TARGETS      acpp target string (default: generic).
  --prefix PATH          AdaptiveCpp installation prefix.
  --jobs N               Parallel build jobs.
  -h, --help             Show this help.

Environment overrides:
  ACPP_REPO, ACPP_BRANCH, TESTS_ACPP_REPO, DEPS_ROOT, ACPP_SRC,
  ACPP_BUILD, ACPP_PREFIX, TESTS_ACPP_SRC, ACPP_TARGETS, JOBS.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-deps)
            INSTALL_DEPS=0
            ;;
        --with-tests-acpp)
            CLONE_TESTS_ACPP=1
            ;;
        --no-build)
            BUILD_CHRIS_LLAMA=0
            ;;
        --no-smoke-test)
            RUN_SMOKE_TEST=0
            ;;
        --no-update)
            UPDATE_SOURCE=0
            ;;
        --branch)
            shift
            ACPP_BRANCH="${1:?missing branch name}"
            ;;
        --targets)
            shift
            ACPP_TARGETS="${1:?missing target string}"
            ;;
        --prefix)
            shift
            ACPP_PREFIX="${1:?missing prefix path}"
            ;;
        --jobs)
            shift
            JOBS="${1:?missing job count}"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

log() {
    printf '\n==> %s\n' "$*"
}

warn() {
    printf 'warning: %s\n' "$*" >&2
}

need_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "error: required command not found: $1" >&2
        exit 1
    fi
}

sudo_cmd() {
    if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        echo "error: root privileges are required to install packages and sudo is unavailable" >&2
        exit 1
    fi
}

install_debian_dependencies() {
    log "Installing build dependencies with apt"
    sudo_cmd apt-get update
    sudo_cmd apt-get install -y \
        build-essential \
        git \
        cmake \
        ninja-build \
        python3 \
        python3-pip \
        pkg-config \
        libboost-all-dev \
        clang \
        llvm-dev \
        libclang-dev \
        libomp-dev \
        lld
}

install_dependencies() {
    if [[ "$INSTALL_DEPS" -eq 0 ]]; then
        return
    fi

    if [[ -r /etc/os-release ]]; then
        # shellcheck disable=SC1091
        source /etc/os-release
    else
        warn "cannot identify the Linux distribution; skipping package installation"
        return
    fi

    case "${ID:-}" in
        ubuntu|debian|linuxmint|pop)
            install_debian_dependencies
            ;;
        *)
            warn "automatic package installation is implemented for Debian/Ubuntu-family systems only"
            warn "install Git, CMake, Boost, Python 3 and a supported official LLVM/Clang release manually"
            ;;
    esac
}

find_clang() {
    local candidate

    if [[ -n "${CXX:-}" && -x "$(command -v "$CXX" 2>/dev/null || true)" ]]; then
        printf '%s\n' "$CXX"
        return
    fi

    for candidate in clang++-21 clang++-20 clang++-19 clang++-18 clang++-17 clang++-16 clang++-15 clang++; do
        if command -v "$candidate" >/dev/null 2>&1; then
            command -v "$candidate"
            return
        fi
    done

    return 1
}

find_clang_c() {
    local cxx="$1"
    local dir name candidate
    dir="$(dirname "$cxx")"
    name="$(basename "$cxx")"
    candidate="$dir/${name/clang++/clang}"

    if [[ -x "$candidate" ]]; then
        printf '%s\n' "$candidate"
        return
    fi

    command -v clang
}

find_llvm_config() {
    local candidate
    for candidate in llvm-config-21 llvm-config-20 llvm-config-19 llvm-config-18 llvm-config-17 llvm-config-16 llvm-config-15 llvm-config; do
        if command -v "$candidate" >/dev/null 2>&1; then
            command -v "$candidate"
            return
        fi
    done
    return 1
}

check_llvm() {
    local llvm_config="$1"
    local version major
    version="$($llvm_config --version)"
    major="${version%%.*}"

    log "Using LLVM $version from $llvm_config"

    if [[ "$major" =~ ^[0-9]+$ ]]; then
        if (( major < 15 )); then
            echo "error: AdaptiveCpp's standard installation requires LLVM >= 15" >&2
            exit 1
        fi
        if (( major > 21 )); then
            warn "LLVM $major is newer than the currently documented supported range (15-21)"
        fi
    fi
}

print_gpu_environment() {
    log "Checking accelerator toolchains"

    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "NVIDIA driver: detected"
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true
        if command -v nvcc >/dev/null 2>&1; then
            echo "CUDA toolkit: detected ($(nvcc --version | tail -n 1))"
        else
            warn "NVIDIA driver is present, but nvcc was not found; CUDA development files may be missing"
        fi
    else
        echo "NVIDIA driver: not detected"
    fi

    if command -v hipcc >/dev/null 2>&1 || [[ -d /opt/rocm ]]; then
        echo "ROCm/HIP: detected"
    else
        echo "ROCm/HIP: not detected"
    fi

    if ldconfig -p 2>/dev/null | grep -q 'libze_loader'; then
        echo "Level Zero loader: detected"
    else
        echo "Level Zero loader: not detected"
    fi
}

clone_or_update() {
    local repo="$1"
    local dest="$2"
    local branch="$3"

    if [[ ! -d "$dest/.git" ]]; then
        log "Cloning $repo"
        git clone --branch "$branch" --single-branch "$repo" "$dest"
        return
    fi

    echo "Using existing clone: $dest"
    if [[ "$UPDATE_SOURCE" -eq 0 ]]; then
        return
    fi

    if [[ -n "$(git -C "$dest" status --porcelain)" ]]; then
        warn "$dest has local changes; leaving it untouched"
        return
    fi

    log "Updating $(basename "$dest")"
    git -C "$dest" fetch origin "$branch"
    git -C "$dest" checkout "$branch"
    git -C "$dest" pull --ff-only origin "$branch"
}

install_dependencies

need_command git
need_command cmake
need_command python3

CXX_COMPILER="$(find_clang || true)"
if [[ -z "$CXX_COMPILER" ]]; then
    echo "error: no supported clang++ compiler was found" >&2
    exit 1
fi
C_COMPILER="$(find_clang_c "$CXX_COMPILER")"
LLVM_CONFIG="$(find_llvm_config || true)"
if [[ -z "$LLVM_CONFIG" ]]; then
    echo "error: llvm-config was not found; install an official LLVM release >= 15" >&2
    exit 1
fi

check_llvm "$LLVM_CONFIG"
print_gpu_environment

mkdir -p "$DEPS_ROOT"
clone_or_update "$ACPP_REPO" "$ACPP_SRC" "$ACPP_BRANCH"

if [[ "$CLONE_TESTS_ACPP" -eq 1 ]]; then
    clone_or_update "$TESTS_ACPP_REPO" "$TESTS_ACPP_SRC" main
fi

LLVM_DIR="$($LLVM_CONFIG --cmakedir)"

log "Configuring AdaptiveCpp from Christian's fork"
echo "source:  $ACPP_SRC"
echo "build:   $ACPP_BUILD"
echo "install: $ACPP_PREFIX"
echo "targets: $ACPP_TARGETS"

cmake -S "$ACPP_SRC" -B "$ACPP_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$ACPP_PREFIX" \
    -DCMAKE_C_COMPILER="$C_COMPILER" \
    -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
    -DLLVM_DIR="$LLVM_DIR" \
    -DACPP_COMPILER_FEATURE_PROFILE=full

log "Building AdaptiveCpp"
cmake --build "$ACPP_BUILD" --parallel "$JOBS"

log "Installing AdaptiveCpp"
cmake --install "$ACPP_BUILD"

ACPP_BIN="$ACPP_PREFIX/bin/acpp"
if [[ ! -x "$ACPP_BIN" ]]; then
    echo "error: AdaptiveCpp installation completed without producing $ACPP_BIN" >&2
    exit 1
fi

export PATH="$ACPP_PREFIX/bin:$PATH"
export CMAKE_PREFIX_PATH="$ACPP_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
if [[ -d "$ACPP_PREFIX/lib" ]]; then
    export LD_LIBRARY_PATH="$ACPP_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ -d "$ACPP_PREFIX/lib64" ]]; then
    export LD_LIBRARY_PATH="$ACPP_PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

log "AdaptiveCpp compiler"
"$ACPP_BIN" --version

if command -v acpp-info >/dev/null 2>&1; then
    log "AdaptiveCpp devices"
    acpp-info || warn "acpp-info returned an error; inspect the selected runtime backend"
fi

if [[ "$RUN_SMOKE_TEST" -eq 1 ]]; then
    log "Compiling a minimal SYCL program"
    mkdir -p "$PROJECT_ROOT/build/tests"
    "$ACPP_BIN" \
        --acpp-targets="$ACPP_TARGETS" \
        -std=c++17 \
        "$PROJECT_ROOT/tests/acpp_smoke.cpp" \
        -o "$PROJECT_ROOT/build/tests/acpp_smoke"

    log "Running the SYCL smoke test"
    "$PROJECT_ROOT/build/tests/acpp_smoke"
fi

if [[ "$BUILD_CHRIS_LLAMA" -eq 1 ]]; then
    log "Building Chris Llama with AdaptiveCpp"
    make -C "$PROJECT_ROOT" clean
    make -C "$PROJECT_ROOT" \
        BACKEND=acpp \
        ACPP="$ACPP_BIN" \
        ACPP_TARGETS="$ACPP_TARGETS" \
        -j"$JOBS"

    log "Devices visible to Chris Llama"
    "$PROJECT_ROOT/build/bin/chris_llama" --list-devices || true
fi

cat <<SUMMARY

AdaptiveCpp setup is complete.

Fork:
  $ACPP_REPO

Installation:
  $ACPP_PREFIX

To use this installation in a new shell:
  source "$PROJECT_ROOT/scripts/activate_adaptivecpp.sh"

Then build Chris Llama with:
  make BACKEND=acpp ACPP_TARGETS="$ACPP_TARGETS"

Related AdaptiveCpp experiments:
  https://github.com/christianrss/tests-acpp
SUMMARY
