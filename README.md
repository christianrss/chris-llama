# Chris Llama

Chris Llama is a small inference runtime used to study how a GPT-style model is
loaded and executed below a high-level ML framework. The current model path is
focused on **Chris-GPT-2**, using GGUF for model storage and an optional
**AdaptiveCpp/SYCL** backend for the large linear projections.

The CPU backend is kept as a deterministic reference implementation. The SYCL
backend uses the same model code and moves the expensive MatVec operations to a
device selected by AdaptiveCpp.

## Related repositories

- Chris-GPT-2: https://github.com/christianrss/chris-gpt-2
- Chris Torch: https://github.com/christianrss/chris-torch
- AdaptiveCpp fork used by this project: https://github.com/christianrss/AdaptiveCpp
- AdaptiveCpp experiments and kernel tests: https://github.com/christianrss/tests-acpp

`christianrss/AdaptiveCpp` is a development fork of the upstream AdaptiveCpp
project. `tests-acpp` is a separate personal testbed for SYCL kernels, compiler
behavior, USM experiments and possible AdaptiveCpp changes; it is not part of
the official AdaptiveCpp project.

## What is implemented

- GGUF v2/v3 parsing with metadata, tensor descriptors, alignment and `mmap`
- GPT-2 byte-level BPE tokenizer loaded from GGUF metadata
- GPT-2 forward pass with learned position embeddings
- LayerNorm, GELU, causal self-attention and residual connections
- per-layer KV cache
- greedy and sampling-based text generation
- support for common GGUF weight types used by this project
- serial CPU reference backend
- AdaptiveCpp/SYCL MatVec backend
- persistent device-side weight cache for the AdaptiveCpp backend
- unit and end-to-end tests using a generated tiny GPT-2 GGUF fixture

The runtime does not contain hard-coded chat replies. Model output comes from the
loaded weights.

## Build layout

Generated files stay under `build/`:

```text
build/
├── obj/
├── bin/
│   └── chris_llama
└── tests/
```

The source tree therefore remains free of `.o` files, binaries and temporary
test fixtures.

## CPU build

```bash
make BACKEND=cpu
```

Run:

```bash
./build/bin/chris_llama model.gguf -p "Machine learning is" -n 64
```

Run the test suite:

```bash
make BACKEND=cpu test
```

## AdaptiveCpp setup

The repository includes a bootstrap script that installs build dependencies on
Debian/Ubuntu-family systems, clones **Christian's AdaptiveCpp fork**, builds and
installs it into the project-local `.deps/` directory, runs a SYCL smoke test,
and builds Chris Llama with that installation.

```bash
./scripts/setup_adaptivecpp.sh
```

The default source is:

```text
https://github.com/christianrss/AdaptiveCpp.git
```

and the default branch is `develop`.

The default compiler target is AdaptiveCpp's portable `generic` flow:

```text
--acpp-targets=generic
```

The script deliberately does **not** install or replace NVIDIA drivers, CUDA, or
ROCm. It detects those stacks and reports what it finds. GPU driver/toolkit
installation remains an explicit system-administration step.

To also clone the companion `tests-acpp` repository:

```bash
./scripts/setup_adaptivecpp.sh --with-tests-acpp
```

After setup, activate the local AdaptiveCpp installation in a new shell with:

```bash
source scripts/activate_adaptivecpp.sh
```

Then build normally:

```bash
make BACKEND=acpp ACPP_TARGETS=generic
```

or use the helper:

```bash
./scripts/build_adaptivecpp.sh
```

List devices visible to the selected backend:

```bash
./build/bin/chris_llama --list-devices
```

See [ADAPTIVECPP.md](ADAPTIVECPP.md) for the full toolchain setup and the exact
behavior of the bootstrap script.

## AdaptiveCpp setup options

```text
--no-deps           skip OS package installation
--with-tests-acpp   clone/update christianrss/tests-acpp
--no-build          install AdaptiveCpp without building Chris Llama
--no-smoke-test     skip the minimal SYCL runtime test
--no-update         keep an existing AdaptiveCpp checkout untouched
--branch NAME       select another branch of the AdaptiveCpp fork
--targets TARGETS   override the acpp target string
--prefix PATH       choose another install prefix
--jobs N            control build parallelism
```

For example:

```bash
./scripts/setup_adaptivecpp.sh \
    --branch develop \
    --targets generic \
    --with-tests-acpp
```

## Running Chris-GPT-2

Inspect a GGUF file:

```bash
./build/bin/chris_llama Chris-GPT-2-124M-F16.gguf --inspect
```

Tokenize a prompt:

```bash
./build/bin/chris_llama \
    Chris-GPT-2-124M-F16.gguf \
    --tokenize "Hello world"
```

Generate text on the CPU backend:

```bash
./build/bin/chris_llama \
    Chris-GPT-2-124M-F16.gguf \
    --device cpu \
    -p "The future of artificial intelligence is" \
    -n 64 \
    --temperature 0.8 \
    --top-k 50 \
    --top-p 0.95
```

With an AdaptiveCpp build, select a GPU with:

```bash
./build/bin/chris_llama \
    Chris-GPT-2-124M-F16.gguf \
    --device gpu \
    -p "The future of artificial intelligence is" \
    -n 64
```

You can also select a device by index or by a substring of its name.

## Backend validation

A useful bring-up check is to compare the deterministic next token from both
backends using the same model and prompt:

```bash
./scripts/compare_cpu_acpp.sh \
    Chris-GPT-2-124M-F16.gguf \
    "Machine learning is"
```

The script builds both backends and expects the greedy next-token ID to match.

## Project structure

```text
backend.h             compute backend C ABI
backend_cpu.c         serial reference MatVec backend
backend_acpp.cpp      AdaptiveCpp/SYCL backend and device-side weight cache
common.h              shared C definitions
gguf.c / gguf.h       GGUF parsing and tensor access
gpt2.c / gpt2.h       GPT-2 model loading and forward pass
main.c                 command-line interface
quant.c / quant.h     GGUF dequantization helpers
sampler.c / sampler.h token sampling
tokenizer.c/.h        GPT-2 byte-level BPE tokenizer

scripts/
  setup_adaptivecpp.sh     clone/build/install the AdaptiveCpp fork
  activate_adaptivecpp.sh  configure the current shell for the local install
  build_adaptivecpp.sh     build Chris Llama with AdaptiveCpp
  compare_cpu_acpp.sh      compare deterministic CPU/SYCL output

tests/
  acpp_smoke.cpp           minimal SYCL/USM runtime test
  generate_tiny_gguf.py    deterministic GPT-2 GGUF fixture generator
  test_backend.c           backend MatVec correctness check
  test_quant.c             quantization/dequantization checks
```

## Current performance model

The AdaptiveCpp backend keeps the large projection weights in device memory.
For each MatVec call, it transfers the small input vector, launches the kernel,
and copies the output vector back to the host. Scratch buffers are reused.

This is intentionally a first GPU architecture rather than the final optimized
one. LayerNorm, attention bookkeeping, activation functions and sampling still
run on the host. A later design can keep the hidden state and KV cache on the
device and fuse more of the Transformer block.

Quantized GGUF weights are currently decoded to FP32 before backend upload. A
future quantized kernel can operate directly on Q4/Q5/Q6 blocks and reduce both
memory use and bandwidth.

## Documentation

- [ADAPTIVECPP.md](ADAPTIVECPP.md) — toolchain setup, local installation and SYCL backend details
- [ARCHITECTURE.md](ARCHITECTURE.md) — runtime and backend boundaries
- [TEST_REPORT.md](TEST_REPORT.md) — validation coverage and limitations

## License

See the repository license for the terms that apply to the source code. Model
weights and third-party projects may have their own licensing terms.
