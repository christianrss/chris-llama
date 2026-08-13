# AdaptiveCpp backend

Chris Llama uses AdaptiveCpp as an optional SYCL compute backend. The runtime
itself remains mostly C; `backend_acpp.cpp` is the C++/SYCL boundary and exports
a small C ABI used by the GPT-2 implementation.

## Repositories

This project defaults to Christian's AdaptiveCpp fork:

https://github.com/christianrss/AdaptiveCpp

The fork tracks the upstream AdaptiveCpp project and is used for experiments,
local changes and possible upstream contributions.

A separate test repository is available at:

https://github.com/christianrss/tests-acpp

`tests-acpp` contains small AdaptiveCpp/SYCL experiments, numerical kernels,
USM tests and compiler/runtime investigations. It is useful when debugging the
toolchain independently from Chris Llama.

Upstream AdaptiveCpp:

https://github.com/AdaptiveCpp/AdaptiveCpp

## Why the setup script installs AdaptiveCpp

AdaptiveCpp must be installed after it is built; using only an uninstalled build
tree is not the supported setup. The project-local bootstrap therefore follows
this sequence:

```text
clone
  ↓
cmake configure
  ↓
cmake build
  ↓
cmake install
  ↓
acpp --version
  ↓
SYCL smoke test
  ↓
Chris Llama build
```

The installation lives under `.deps/adaptivecpp` by default and does not require
installing AdaptiveCpp system-wide.

## One-command bootstrap

On a Debian/Ubuntu-family Linux system:

```bash
./scripts/setup_adaptivecpp.sh
```

The script:

1. installs base build dependencies with `apt`;
2. finds an official LLVM/Clang toolchain;
3. checks the LLVM version;
4. reports NVIDIA, CUDA, ROCm/HIP and Level Zero availability;
5. clones `https://github.com/christianrss/AdaptiveCpp.git`;
6. configures a Release build with the full compiler feature profile;
7. builds and installs AdaptiveCpp into `.deps/adaptivecpp`;
8. runs `acpp --version` and `acpp-info` when available;
9. compiles and runs `tests/acpp_smoke.cpp`;
10. builds Chris Llama with `BACKEND=acpp`;
11. prints devices visible through the runtime.

The script does not install GPU drivers, CUDA, or ROCm. Those packages are tied
to the host GPU and driver configuration and should not be replaced by a project
bootstrap script.

## Dependencies installed on Debian/Ubuntu

The bootstrap installs the base development packages required for a standard
AdaptiveCpp source build:

```text
build-essential
git
cmake
ninja-build
python3
python3-pip
pkg-config
libboost-all-dev
clang
llvm-dev
libclang-dev
libomp-dev
lld
```

AdaptiveCpp's current standard installation requires an official LLVM release
of at least version 15. The bootstrap warns when the detected LLVM is newer than
the currently documented supported range.

## Local directory layout

```text
.deps/
├── AdaptiveCpp/          source checkout from christianrss/AdaptiveCpp
├── AdaptiveCpp-build/    CMake build tree
├── adaptivecpp/          installed compiler/runtime
└── tests-acpp/           optional companion checkout
```

`.deps/` is ignored by Git.

## Activate the installation

The setup script cannot permanently change the environment of its parent shell.
In a new terminal, run:

```bash
source scripts/activate_adaptivecpp.sh
```

This adds the local installation to `PATH`, `CMAKE_PREFIX_PATH`, and the dynamic
library search path when needed.

Verify it with:

```bash
acpp --version
acpp-info
```

## Generic target

Chris Llama defaults to:

```text
--acpp-targets=generic
```

AdaptiveCpp's generic single-pass flow embeds backend-independent device IR and
lowers it for the selected device at runtime. This makes it a useful default for
a project intended to move between CPU and GPU systems.

Build with:

```bash
make BACKEND=acpp ACPP_TARGETS=generic
```

The executable is written to:

```text
build/bin/chris_llama
```

## NVIDIA systems

The bootstrap detects `nvidia-smi` and `nvcc` separately. A working driver does
not necessarily imply that the CUDA development toolkit is installed.

For the generic AdaptiveCpp path on NVIDIA, the installed LLVM toolchain also
needs NVPTX support and the appropriate AdaptiveCpp runtime backend must be
available. The setup script reports the detected environment but leaves driver
and CUDA installation to the machine owner.

## AMD systems

The script detects `hipcc` and `/opt/rocm`. ROCm installation is deliberately
outside the bootstrap because the correct package set depends on the GPU,
distribution and host driver configuration.

## Intel / Level Zero systems

When available, the script reports the Level Zero loader. Intel GPU execution
also depends on the host's Level Zero/OpenCL runtime stack.

## SYCL smoke test

`tests/acpp_smoke.cpp` is intentionally small. It creates a default SYCL queue,
allocates USM shared memory, launches a kernel, waits for completion, verifies
the result and prints the selected device name.

The bootstrap compiles it with the same target string used for Chris Llama:

```bash
acpp --acpp-targets=generic -std=c++17 tests/acpp_smoke.cpp -o build/tests/acpp_smoke
./build/tests/acpp_smoke
```

This catches basic compiler/runtime problems before the larger runtime is built.

## Weight residency

`backend_acpp.cpp` uploads model projection weights once and keeps them in
`sycl::malloc_device` allocations. Host pointers are used as cache keys so
repeated MatVec calls reuse the same device allocation.

At inference time the current path is:

```text
host input vector
      ↓
USM device scratch
      ↓
SYCL MatVec using resident weights
      ↓
device output scratch
      ↓
host output vector
```

This avoids copying the full weight matrix over PCIe for every generated token.

## CPU reference backend

The serial C backend is intentionally retained even when AdaptiveCpp is
available. It provides a simple result to compare against when bringing up a new
GPU, compiler version or kernel implementation.

Use:

```bash
make BACKEND=cpu test
```

and compare against AdaptiveCpp with:

```bash
./scripts/compare_cpu_acpp.sh model.gguf "prompt"
```

## Using tests-acpp

To clone the companion testbed during setup:

```bash
./scripts/setup_adaptivecpp.sh --with-tests-acpp
```

The checkout is placed at:

```text
.deps/tests-acpp
```

You can also use it independently from Chris Llama:

```bash
git clone https://github.com/christianrss/tests-acpp.git
```

That repository is the better place for isolated kernel experiments or compiler
reproduction cases that do not belong in the inference runtime itself.
