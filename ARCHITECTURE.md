# Architecture

Chris Llama separates model semantics from compute execution. GGUF parsing,
tokenization, GPT-2 control flow, KV-cache management and sampling are shared by
both backends.

```text
                         Chris Llama
                              │
                 GGUF + tokenizer + GPT-2
                              │
                    chris_backend_matvec()
                              │
                ┌─────────────┴─────────────┐
                │                           │
        backend_cpu.c               backend_acpp.cpp
        serial C MatVec             AdaptiveCpp / SYCL
                │                           │
               CPU                    CPU / GPU device
```

## Backend boundary

`backend.h` is a C ABI. The GPT-2 implementation only needs to know how to:

- create and destroy a backend;
- list/select devices;
- preload read-only model data;
- execute a matrix-vector product.

The AdaptiveCpp implementation can therefore use C++ objects, RAII, `sycl::queue`
and USM internally without forcing the rest of the runtime to become C++.

## Model load

The GGUF loader maps the file, reads metadata and tensor descriptors, and exposes
weights to the GPT-2 loader. Supported quantized blocks are decoded to FP32 by
the current reference path.

When an AdaptiveCpp backend is attached, the large projection matrices are
preloaded into device memory.

## Token generation

For each token, the model performs:

```text
token embedding + position embedding
              ↓
         Transformer block
              │
              ├─ LayerNorm
              ├─ QKV projection ---------> backend MatVec
              ├─ causal attention / KV cache
              ├─ output projection ------> backend MatVec
              ├─ residual
              ├─ LayerNorm
              ├─ FFN up projection ------> backend MatVec
              ├─ GELU
              ├─ FFN down projection ----> backend MatVec
              └─ residual
              ↓
         final LayerNorm
              ↓
         LM-head MatVec -----------------> backend MatVec
              ↓
             logits
              ↓
            sampler
```

The current SYCL design accelerates the large linear operations first because
they dominate the amount of model arithmetic while keeping the implementation
easy to compare with the CPU reference.

## Device memory

The AdaptiveCpp backend keeps read-only weights resident on the selected device.
Scratch input/output allocations grow only when a larger vector is needed and
are then reused.

A future version can keep the hidden state and KV cache on the device as well,
which would remove several host/device transfers and make fusion of complete
Transformer blocks possible.
