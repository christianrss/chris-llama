# Test report

The repository contains a deterministic CPU reference path and tests for the
parts of the runtime that can be validated without a full Chris-GPT checkpoint.

## CPU tests

`make BACKEND=cpu test` checks:

- GGUF quantization/dequantization helpers;
- backend MatVec output against hand-computed values;
- generation of a deterministic tiny GPT-2 GGUF fixture;
- GGUF parsing;
- byte-level BPE tokenization;
- BPE merge behavior;
- full GPT-2 forward execution;
- deterministic greedy next-token output against a Python-generated reference.

The expected token for the bundled fixture is stored in `tests/expected.txt`.

## AdaptiveCpp tests

The bootstrap adds a separate `tests/acpp_smoke.cpp` check. It validates that the
local `acpp` compiler can build a SYCL program, create a queue, allocate USM,
launch a kernel and read back the correct result.

On a machine with AdaptiveCpp and the desired GPU runtime installed, run:

```bash
./scripts/setup_adaptivecpp.sh
make BACKEND=acpp test
```

For a real GGUF model, compare deterministic output between backends with:

```bash
./scripts/compare_cpu_acpp.sh model.gguf "prompt"
```

## Environment limitation of this archive

The archive was prepared in an environment without an installed AdaptiveCpp GPU
toolchain. The CPU test suite can be executed here, but the SYCL backend must be
compiled and run on a host where AdaptiveCpp and the target runtime are
available.

The bootstrap script is included specifically to make that machine setup
repeatable from Christian's AdaptiveCpp fork.

## Packaging validation

After the bootstrap and build-directory changes, the CPU suite completed with:

```text
quant tests passed
backend=cpu device=CPU reference matvec OK
expected next token=141
actual next token=141
All tests passed (CPU reference).
```

All shell helpers were also parsed successfully with `bash -n`.
