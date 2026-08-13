# Chris Llama

Chris Llama is an experimental low-level inference runtime for GPT-style language models.

The current implementation focuses on **Chris-GPT-2** models stored in GGUF format and provides CPU execution together with an optional **AdaptiveCpp/SYCL** backend for heterogeneous compute.

The runtime includes GGUF parsing, GPT-2 byte-level BPE tokenization, autoregressive inference, KV caching, sampling, quantization support, and device-side execution of large linear projections.

## Features

* GGUF v2/v3 metadata parsing
* GGUF tensor descriptor parsing
* memory-mapped model access
* GPT-2 byte-level BPE tokenizer
* GPT-2 autoregressive inference
* learned positional embeddings
* causal multi-head self-attention
* per-layer KV cache
* LayerNorm
* GELU
* residual connections
* greedy decoding
* temperature sampling
* top-k sampling
* top-p sampling
* repetition penalty
* GGUF weight dequantization
* deterministic CPU reference backend
* AdaptiveCpp/SYCL compute backend
* persistent device-side weight storage
* runtime device selection
* synthetic GGUF fixtures for deterministic validation
* CPU/SYCL backend comparison tools

## Supported Model Path

The primary reference model is:

**Chris-GPT-2 124M**

Transformers / SafeTensors:

https://huggingface.co/christianrss/chris-gpt-2-124m

GGUF:

https://huggingface.co/christianrss/chris-gpt-2-124m-GGUF

Training repository:

https://github.com/christianrss/chris-gpt-2

Chris-GPT-2 is a GPT-2 124M-scale language model pretrained from scratch on approximately 10 billion FineWeb-Edu tokens.

The GGUF export is used as the main integration workload for Chris Llama.

## Model Execution

The current GPT-2 execution path is:

```text
GGUF file
   │
   ▼
metadata + tensors
   │
   ▼
GPT-2 BPE tokenizer
   │
   ▼
token embeddings
   │
   ▼
position embeddings
   │
   ▼
Transformer blocks
   │
   ├── LayerNorm
   ├── QKV projection
   ├── causal attention
   ├── output projection
   ├── residual
   ├── LayerNorm
   ├── feed-forward projection
   ├── GELU
   ├── output projection
   └── residual
   │
   ▼
final LayerNorm
   │
   ▼
language-model head
   │
   ▼
logits
   │
   ▼
sampler
   │
   ▼
next token
```

Inference is performed directly from tensors loaded from the model.

## Compute Backends

Two compute backends are currently available.

### CPU

The CPU backend provides the deterministic reference implementation.

```text
BACKEND=cpu
```

It is used for correctness validation and as a fallback when AdaptiveCpp is unavailable.

### AdaptiveCpp / SYCL

The AdaptiveCpp backend provides heterogeneous execution through SYCL.

```text
BACKEND=acpp
```

The model execution path remains shared between both backends. Large linear projections are dispatched through the backend interface.

The AdaptiveCpp implementation currently accelerates MatVec operations and maintains projection weights in device memory.

## Backend Interface

The model implementation calls a backend-neutral compute API.

```text
GPT-2 model
    │
    ▼
backend interface
    │
    ├── backend_cpu.c
    │
    └── backend_acpp.cpp
             │
             ▼
        AdaptiveCpp/SYCL
```

The C model code remains independent from SYCL-specific types.

The AdaptiveCpp backend is implemented in C++ and exposes a C-compatible interface to the rest of the runtime.

## Build Layout

Generated build artifacts are stored under `build/`.

```text
build/
├── obj/
│   ├── main.o
│   ├── gguf.o
│   ├── quant.o
│   ├── tokenizer.o
│   ├── gpt2.o
│   ├── sampler.o
│   └── backend_*.o
│
├── bin/
│   └── chris_llama
│
└── tests/
    ├── test_quant
    ├── test_backend
    ├── actual.txt
    └── tiny-gpt2.gguf
```

Third-party local dependencies are stored separately:

```text
.deps/
```

Typical AdaptiveCpp development setup:

```text
.deps/
├── AdaptiveCpp/
├── AdaptiveCpp-build/
├── adaptivecpp/
└── tests-acpp/
```

Both `build/` and `.deps/` are excluded from version control.

## Requirements

### CPU Build

Required tools:

* GCC or compatible C11 compiler
* GNU Make
* Python 3 for test fixture generation

### AdaptiveCpp Build

Additional requirements:

* CMake
* C++17 compiler
* LLVM/Clang as required by the selected AdaptiveCpp configuration
* AdaptiveCpp
* compatible device runtime
* vendor GPU stack when targeting GPU hardware

GPU-specific driver stacks are treated as host dependencies.

Examples include:

```text
NVIDIA GPU
    NVIDIA driver
    CUDA-compatible environment

AMD GPU
    ROCm-compatible environment

Intel GPU
    Level Zero / supported Intel runtime
```

The project bootstrap does not install or modify GPU drivers.

## CPU Build

Build the reference backend:

```bash
make BACKEND=cpu
```

The executable is generated at:

```text
build/bin/chris_llama
```

Run:

```bash
./build/bin/chris_llama model.gguf \
    -p "Machine learning is" \
    -n 64
```

Clean generated artifacts:

```bash
make clean
```

## Automatic Backend Selection

The default build configuration uses:

```text
BACKEND=auto
```

When `acpp` is available in `PATH`, the AdaptiveCpp backend is selected.

Otherwise the CPU reference backend is used.

Equivalent command:

```bash
make
```

## AdaptiveCpp Toolchain

Chris Llama uses the following AdaptiveCpp fork by default:

https://github.com/christianrss/AdaptiveCpp

The repository contains a complete local bootstrap script:

```bash
./scripts/setup_adaptivecpp.sh
```

The bootstrap installs AdaptiveCpp inside the project dependency directory instead of requiring a global installation.

Default source:

```text
https://github.com/christianrss/AdaptiveCpp.git
```

Default branch:

```text
develop
```

Default installation prefix:

```text
.deps/adaptivecpp
```

Default AdaptiveCpp target:

```text
generic
```

## AdaptiveCpp Bootstrap

Run:

```bash
./scripts/setup_adaptivecpp.sh
```

The bootstrap performs:

```text
host dependency check
        │
        ▼
AdaptiveCpp clone/update
        │
        ▼
CMake configure
        │
        ▼
AdaptiveCpp build
        │
        ▼
local installation
        │
        ▼
acpp validation
        │
        ▼
SYCL/USM smoke test
        │
        ▼
Chris Llama AdaptiveCpp build
        │
        ▼
device enumeration
```

On supported Debian/Ubuntu-family systems, the script can install general build dependencies.

GPU drivers, CUDA, ROCm, and other vendor-level system components are not installed or replaced.

## Bootstrap Options

```text
--no-deps
    Skip operating-system package installation.

--with-tests-acpp
    Clone or update christianrss/tests-acpp.

--no-build
    Install AdaptiveCpp without building Chris Llama.

--no-smoke-test
    Skip the minimal SYCL/USM runtime validation.

--no-update
    Preserve an existing AdaptiveCpp checkout without updating it.

--branch NAME
    Select the AdaptiveCpp branch.

--targets TARGETS
    Override the AdaptiveCpp target string.

--prefix PATH
    Override the local AdaptiveCpp installation prefix.

--jobs N
    Set build parallelism.
```

Example:

```bash
./scripts/setup_adaptivecpp.sh \
    --branch develop \
    --targets generic \
    --with-tests-acpp
```

## Activate the Local AdaptiveCpp Installation

The local installation can be activated with:

```bash
source scripts/activate_adaptivecpp.sh
```

This configures the current shell to use the AdaptiveCpp installation under `.deps/`.

Verify:

```bash
acpp --version
```

When available:

```bash
acpp-info
```

can be used to inspect the AdaptiveCpp environment.

## AdaptiveCpp Build

After toolchain activation:

```bash
make BACKEND=acpp ACPP_TARGETS=generic
```

The helper script provides the same project build path:

```bash
./scripts/build_adaptivecpp.sh
```

Executable:

```text
build/bin/chris_llama
```

## Device Enumeration

For an AdaptiveCpp build:

```bash
./build/bin/chris_llama --list-devices
```

The runtime supports device selection through the `--device` option.

Examples:

```bash
--device auto
```

```bash
--device cpu
```

```bash
--device gpu
```

Device selection can also use a numeric index or a name substring when supported by the backend.

## Running Chris-GPT-2

Reference GGUF:

```text
Chris-GPT-2-124M-F16.gguf
```

### Inspect Model Metadata

```bash
./build/bin/chris_llama \
    Chris-GPT-2-124M-F16.gguf \
    --inspect
```

### List Tensors

```bash
./build/bin/chris_llama \
    Chris-GPT-2-124M-F16.gguf \
    --list-tensors
```

### Tokenization

```bash
./build/bin/chris_llama \
    Chris-GPT-2-124M-F16.gguf \
    --tokenize "Hello world"
```

### Deterministic Next Token

```bash
./build/bin/chris_llama \
    Chris-GPT-2-124M-F16.gguf \
    -p "Machine learning is" \
    --next-token-id \
    --temperature 0
```

### CPU Generation

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

### AdaptiveCpp GPU Generation

```bash
./build/bin/chris_llama \
    Chris-GPT-2-124M-F16.gguf \
    --device gpu \
    -p "The future of artificial intelligence is" \
    -n 64 \
    --temperature 0.8 \
    --top-k 50 \
    --top-p 0.95
```

## Sampling

The runtime supports deterministic and stochastic generation.

### Greedy

```bash
--temperature 0
```

### Temperature

```bash
--temperature 0.8
```

### Top-K

```bash
--top-k 50
```

### Top-P

```bash
--top-p 0.95
```

The options can be combined:

```bash
./build/bin/chris_llama model.gguf \
    -p "The future of artificial intelligence is" \
    -n 64 \
    --temperature 0.8 \
    --top-k 50 \
    --top-p 0.95
```

## Tests

Run the CPU validation suite:

```bash
make BACKEND=cpu test
```

The suite covers:

* GGUF parsing
* tokenizer behavior
* quantization/dequantization
* backend MatVec correctness
* GPT-2 model loading
* deterministic forward execution
* deterministic next-token validation

Test artifacts are generated under:

```text
build/tests/
```

## Synthetic GGUF Fixture

The test suite generates a small deterministic GPT-2 GGUF model:

```text
build/tests/tiny-gpt2.gguf
```

Generator:

```text
tests/generate_tiny_gguf.py
```

The fixture provides controlled tensors and expected output values for testing parser and inference changes without depending on the full Chris-GPT-2 checkpoint.

## AdaptiveCpp Smoke Test

The AdaptiveCpp bootstrap includes a minimal SYCL/USM test:

```text
tests/acpp_smoke.cpp
```

The smoke test verifies:

* AdaptiveCpp compilation
* SYCL queue creation
* device visibility
* USM allocation
* kernel execution
* result transfer

It is intended as a toolchain validation step before building the full runtime.

## Backend Validation

CPU and AdaptiveCpp results can be compared with:

```bash
./scripts/compare_cpu_acpp.sh \
    Chris-GPT-2-124M-F16.gguf \
    "Machine learning is"
```

The comparison uses deterministic next-token generation.

Expected validation condition:

```text
CPU next token == AdaptiveCpp next token
```

A difference indicates a backend numerical or execution issue requiring investigation.

## AdaptiveCpp Backend Architecture

Large projection weights are transferred to device memory during backend initialization and retained there for subsequent inference operations.

Current MatVec execution:

```text
host activation
      │
      ▼
device input buffer
      │
      ▼
resident device weights
      │
      ▼
SYCL MatVec kernel
      │
      ▼
device output buffer
      │
      ▼
host activation
```

Scratch allocations are reused between calls.

This avoids repeatedly transferring large projection matrices across the host/device boundary.

Current offloaded operations include the large GPT-2 linear projections used for:

* QKV projection
* attention output projection
* feed-forward expansion
* feed-forward projection
* language-model head

## Memory Model

The AdaptiveCpp backend uses USM-based device allocations for persistent model data.

Conceptually:

```text
GGUF tensor
    │
    ▼
host representation
    │
    ▼
device allocation
    │
    ▼
persistent backend weight
```

Projection weights remain resident on the device after backend attachment.

Small activation vectors are transferred as required by the current execution design.

## Quantized Weights

GGUF quantized weight formats are decoded before execution by the current compute backend.

Current path:

```text
GGUF quantized tensor
        │
        ▼
dequantization
        │
        ▼
FP32 representation
        │
        ▼
backend upload
        │
        ▼
MatVec
```

This implementation favors correctness and backend simplicity over memory efficiency.

Direct quantized MatVec kernels are not currently implemented.

## Current Limitations

The AdaptiveCpp backend currently has the following architectural limitations:

* hidden states are not permanently device-resident;
* KV cache remains host-managed;
* LayerNorm executes on the host;
* GELU executes on the host;
* attention control flow executes on the host;
* sampling executes on the host;
* Transformer blocks are not fused;
* activation vectors cross the host/device boundary between selected operations;
* quantized matrix-vector kernels are not implemented;
* quantized weights are expanded before backend execution.

The CPU implementation remains the numerical reference path.

## Repository Structure

```text
.
├── Makefile
├── README.md
├── ADAPTIVECPP.md
├── ARCHITECTURE.md
├── TEST_REPORT.md
├── BUILD_INFO.txt
│
├── main.c
├── common.h
│
├── gguf.c
├── gguf.h
│
├── gpt2.c
├── gpt2.h
│
├── tokenizer.c
├── tokenizer.h
│
├── sampler.c
├── sampler.h
│
├── quant.c
├── quant.h
│
├── backend.h
├── backend_cpu.c
└── backend_acpp.cpp
```

Scripts:

```text
scripts/
├── setup_adaptivecpp.sh
├── activate_adaptivecpp.sh
├── build_adaptivecpp.sh
└── compare_cpu_acpp.sh
```

Tests:

```text
tests/
├── acpp_smoke.cpp
├── generate_tiny_gguf.py
├── test_backend.c
├── test_quant.c
└── expected.txt
```

Generated directories:

```text
build/
.deps/
```

## Source Responsibilities

### `gguf.c` / `gguf.h`

GGUF file access, metadata parsing, tensor descriptors, alignment, and mapped tensor storage.

### `tokenizer.c` / `tokenizer.h`

GPT-2 byte-level BPE tokenization based on vocabulary and merges stored in GGUF metadata.

### `gpt2.c` / `gpt2.h`

GPT-2 model loading, Transformer execution, attention, KV-cache handling, and language-model projection.

### `sampler.c` / `sampler.h`

Greedy and stochastic token selection.

### `quant.c` / `quant.h`

GGUF quantized tensor decoding.

### `backend.h`

Compute backend interface shared by model execution.

### `backend_cpu.c`

Serial reference MatVec backend.

### `backend_acpp.cpp`

AdaptiveCpp/SYCL backend, device selection, USM allocation, persistent device weight storage, and MatVec execution.

### `main.c`

Command-line interface and runtime configuration.

## Related Projects

### Chris-GPT-2

https://github.com/christianrss/chris-gpt-2

GPT-2 124M-scale model implementation and pretraining project.

Public model:

https://huggingface.co/christianrss/chris-gpt-2-124m

GGUF model:

https://huggingface.co/christianrss/chris-gpt-2-124m-GGUF

### Chris Torch

https://github.com/christianrss/chris-torch

Experimental machine-learning framework covering tensors, autograd, modules, optimizers, and native compute backends.

### AdaptiveCpp Fork

https://github.com/christianrss/AdaptiveCpp

AdaptiveCpp fork used by the local bootstrap and heterogeneous compute development environment.

### tests-acpp

https://github.com/christianrss/tests-acpp

Independent AdaptiveCpp/SYCL validation repository containing kernel, USM, compiler, runtime, and backend experiments.

`tests-acpp` is not a runtime dependency of Chris Llama.

## Project Relationship

```text
                  Chris-GPT-2
                  model weights
                       │
                       ▼
                     GGUF
                       │
                       ▼
                  Chris Llama
                       │
                compute backend
                       │
              ┌────────┴────────┐
              ▼                 ▼
             CPU          AdaptiveCpp
                                │
                               SYCL
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
                   CPU       NVIDIA        AMD
                               GPU          GPU
```

Chris Torch and `tests-acpp` provide separate environments for framework and heterogeneous-compute experiments.

## Documentation

Additional documentation:

* [ADAPTIVECPP.md](ADAPTIVECPP.md) — AdaptiveCpp toolchain, bootstrap, installation, and backend configuration
* [ARCHITECTURE.md](ARCHITECTURE.md) — runtime components and backend boundaries
* [TEST_REPORT.md](TEST_REPORT.md) — validation coverage and known limitations
* [BUILD_INFO.txt](BUILD_INFO.txt) — build notes

## Model Research

Chris-GPT-2 is documented in:

**A Reproducible 10-Billion-Token Pretraining Run of GPT-2 124M**

ResearchGate:

https://www.researchgate.net/publication/412096883_A_Reproducible_10-Billion-Token_Pretraining_Run_of_GPT-2_124M

Original training checkpoints:

https://drive.google.com/drive/folders/1vD9DTvZKRBJjvY_ALqJWeXrVHYnZKOH6?usp=sharing

## References

* Radford, A. et al. *Language Models are Unsupervised Multitask Learners*. OpenAI, 2019.
* Vaswani, A. et al. *Attention Is All You Need*. 2017.
* GGUF / GGML
* llama.cpp
* AdaptiveCpp
* SYCL
* Hugging Face Transformers
* FineWeb-Edu

## License

See the repository license for the terms applying to Chris Llama source code.

Model weights, datasets, AdaptiveCpp, and other third-party components remain subject to their respective licenses.
