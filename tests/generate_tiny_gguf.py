#!/usr/bin/env python3
"""Build a tiny GPT-2 GGUF fixture and compute its reference next token.

The fixture uses the same tensor names as GPT-2 GGUF files, but keeps the model
small enough to exercise the full C inference path in a few seconds. The file is
generated locally so the test suite does not need to ship third-party weights.
"""

import math
import os
import random
import struct
import sys
from pathlib import Path


GGUF_UINT32 = 4
GGUF_FLOAT32 = 6
GGUF_STRING = 8
GGUF_ARRAY = 9

GGML_TYPE_F32 = 0
GGUF_ALIGNMENT = 32

EMBEDDING_SIZE = 8
HEAD_COUNT = 2
LAYER_COUNT = 2
FFN_SIZE = 16
CONTEXT_LENGTH = 32

RANDOM_SEED = 12345


random.seed(RANDOM_SEED)


def bytes_to_unicode() -> dict[int, str]:
    """Return the reversible byte-to-Unicode table used by GPT-2 BPE."""
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(161, 173))
        + list(range(174, 256))
    )
    unicode_values = byte_values[:]

    next_codepoint = 256
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            unicode_values.append(next_codepoint)
            next_codepoint += 1

    return {
        byte: chr(codepoint)
        for byte, codepoint in zip(byte_values, unicode_values)
    }


BYTE_ENCODER = bytes_to_unicode()
TOKENS = [BYTE_ENCODER[i] for i in range(256)] + [
    "<|endoftext|>",
    "He",
    "Hel",
    "Hell",
    "Hello",
    "Ġw",
    "Ġwo",
    "Ġwor",
    "Ġworl",
    "Ġworld",
]
MERGES = [
    "H e",
    "He l",
    "Hel l",
    "Hell o",
    "Ġ w",
    "Ġw o",
    "Ġwo r",
    "Ġwor l",
    "Ġworl d",
]
VOCAB_SIZE = len(TOKENS)


# Tensor entries are stored as (name, GGML dimensions, physical F32 values).
tensors: list[tuple[str, list[int], list[float]]] = []
weights: dict[str, tuple[list[int], list[float]]] = {}


def add_tensor(name: str, dims: list[int], values: list[float]) -> None:
    """Register one tensor for both GGUF output and the Python reference model."""
    if math.prod(dims) != len(values):
        raise ValueError(f"{name}: shape {dims} does not match {len(values)} values")

    values = [float(value) for value in values]
    tensors.append((name, dims, values))
    weights[name] = (dims, values)


def random_values(count: int, scale: float = 0.05) -> list[float]:
    return [(random.random() * 2.0 - 1.0) * scale for _ in range(count)]


def ones(count: int) -> list[float]:
    return [1.0] * count


def zeros(count: int) -> list[float]:
    return [0.0] * count


def build_tensors() -> None:
    """Create deterministic weights for a very small GPT-2 model."""
    add_tensor(
        "token_embd.weight",
        [EMBEDDING_SIZE, VOCAB_SIZE],
        random_values(EMBEDDING_SIZE * VOCAB_SIZE, 0.08),
    )
    add_tensor(
        "position_embd.weight",
        [EMBEDDING_SIZE, CONTEXT_LENGTH],
        random_values(EMBEDDING_SIZE * CONTEXT_LENGTH, 0.03),
    )

    for layer in range(LAYER_COUNT):
        prefix = f"blk.{layer}."

        add_tensor(prefix + "attn_norm.weight", [EMBEDDING_SIZE], ones(EMBEDDING_SIZE))
        add_tensor(prefix + "attn_norm.bias", [EMBEDDING_SIZE], zeros(EMBEDDING_SIZE))
        add_tensor(
            prefix + "attn_qkv.weight",
            [EMBEDDING_SIZE, 3 * EMBEDDING_SIZE],
            random_values(EMBEDDING_SIZE * 3 * EMBEDDING_SIZE, 0.07),
        )
        add_tensor(
            prefix + "attn_qkv.bias",
            [3 * EMBEDDING_SIZE],
            random_values(3 * EMBEDDING_SIZE, 0.01),
        )
        add_tensor(
            prefix + "attn_output.weight",
            [EMBEDDING_SIZE, EMBEDDING_SIZE],
            random_values(EMBEDDING_SIZE * EMBEDDING_SIZE, 0.07),
        )
        add_tensor(
            prefix + "attn_output.bias",
            [EMBEDDING_SIZE],
            random_values(EMBEDDING_SIZE, 0.01),
        )

        add_tensor(prefix + "ffn_norm.weight", [EMBEDDING_SIZE], ones(EMBEDDING_SIZE))
        add_tensor(prefix + "ffn_norm.bias", [EMBEDDING_SIZE], zeros(EMBEDDING_SIZE))
        add_tensor(
            prefix + "ffn_up.weight",
            [EMBEDDING_SIZE, FFN_SIZE],
            random_values(EMBEDDING_SIZE * FFN_SIZE, 0.07),
        )
        add_tensor(
            prefix + "ffn_up.bias",
            [FFN_SIZE],
            random_values(FFN_SIZE, 0.01),
        )
        add_tensor(
            prefix + "ffn_down.weight",
            [FFN_SIZE, EMBEDDING_SIZE],
            random_values(FFN_SIZE * EMBEDDING_SIZE, 0.07),
        )
        add_tensor(
            prefix + "ffn_down.bias",
            [EMBEDDING_SIZE],
            random_values(EMBEDDING_SIZE, 0.01),
        )

    add_tensor("output_norm.weight", [EMBEDDING_SIZE], ones(EMBEDDING_SIZE))
    add_tensor("output_norm.bias", [EMBEDDING_SIZE], zeros(EMBEDDING_SIZE))

    # output.weight is intentionally omitted. GPT-2 ties the LM head to the
    # token embedding matrix, and this fixture verifies that fallback path.


METADATA = [
    ("general.architecture", GGUF_STRING, "gpt2"),
    ("general.name", GGUF_STRING, "tiny-chris-gpt2-test"),
    ("general.alignment", GGUF_UINT32, GGUF_ALIGNMENT),
    ("gpt2.context_length", GGUF_UINT32, CONTEXT_LENGTH),
    ("gpt2.embedding_length", GGUF_UINT32, EMBEDDING_SIZE),
    ("gpt2.block_count", GGUF_UINT32, LAYER_COUNT),
    ("gpt2.feed_forward_length", GGUF_UINT32, FFN_SIZE),
    ("gpt2.attention.head_count", GGUF_UINT32, HEAD_COUNT),
    ("gpt2.attention.layer_norm_epsilon", GGUF_FLOAT32, 1e-5),
    ("tokenizer.ggml.model", GGUF_STRING, "gpt2"),
    ("tokenizer.ggml.tokens", GGUF_ARRAY, (GGUF_STRING, TOKENS)),
    ("tokenizer.ggml.merges", GGUF_ARRAY, (GGUF_STRING, MERGES)),
    ("tokenizer.ggml.bos_token_id", GGUF_UINT32, 256),
    ("tokenizer.ggml.eos_token_id", GGUF_UINT32, 256),
]


def write_string(file, value: str) -> None:
    encoded = value.encode("utf-8")
    file.write(struct.pack("<Q", len(encoded)))
    file.write(encoded)


def write_metadata(file, key: str, value_type: int, value) -> None:
    write_string(file, key)
    file.write(struct.pack("<I", value_type))

    if value_type == GGUF_STRING:
        write_string(file, value)
        return

    if value_type == GGUF_UINT32:
        file.write(struct.pack("<I", value))
        return

    if value_type == GGUF_FLOAT32:
        file.write(struct.pack("<f", value))
        return

    if value_type == GGUF_ARRAY:
        element_type, entries = value
        file.write(struct.pack("<IQ", element_type, len(entries)))
        if element_type != GGUF_STRING:
            raise NotImplementedError("The fixture only needs string arrays")
        for entry in entries:
            write_string(file, entry)
        return

    raise NotImplementedError(f"Unsupported fixture metadata type: {value_type}")


def align_up(value: int, alignment: int = GGUF_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def write_gguf(path: Path) -> None:
    """Write the fixture as a GGUF v3 file."""
    offsets: list[int] = []
    current_offset = 0

    for _, _, values in tensors:
        current_offset = align_up(current_offset)
        offsets.append(current_offset)
        current_offset += 4 * len(values)

    with path.open("wb") as file:
        file.write(b"GGUF")
        file.write(struct.pack("<IQQ", 3, len(tensors), len(METADATA)))

        for item in METADATA:
            write_metadata(file, *item)

        for (name, dims, _), offset in zip(tensors, offsets):
            write_string(file, name)
            file.write(struct.pack("<I", len(dims)))
            for dim in dims:
                file.write(struct.pack("<Q", dim))
            file.write(struct.pack("<IQ", GGML_TYPE_F32, offset))

        while file.tell() % GGUF_ALIGNMENT:
            file.write(b"\0")

        data_start = file.tell()
        for (_, _, values), offset in zip(tensors, offsets):
            while file.tell() < data_start + offset:
                file.write(b"\0")
            file.write(struct.pack(f"<{len(values)}f", *values))


def matvec(name: str, x: list[float]) -> list[float]:
    """Reference row-major matrix-vector product using one named fixture tensor."""
    dims, matrix = weights[name]
    input_size = dims[0]
    output_size = dims[1] if len(dims) > 1 else 1

    if len(x) != input_size:
        raise ValueError(f"{name}: expected vector length {input_size}, got {len(x)}")

    return [
        sum(matrix[row * input_size + col] * x[col] for col in range(input_size))
        for row in range(output_size)
    ]


def add_bias(values: list[float], bias_name: str) -> list[float]:
    bias = weights[bias_name][1]
    return [value + offset for value, offset in zip(values, bias)]


def layer_norm(x: list[float], weight_name: str, bias_name: str) -> list[float]:
    mean = sum(x) / len(x)
    variance = sum((value - mean) ** 2 for value in x) / len(x)
    inv_std = 1.0 / math.sqrt(variance + 1e-5)

    scale = weights[weight_name][1]
    bias = weights[bias_name][1]
    return [
        (x[i] - mean) * inv_std * scale[i] + bias[i]
        for i in range(len(x))
    ]


def gelu(value: float) -> float:
    cubic = value * value * value
    inner = math.sqrt(2.0 / math.pi) * (value + 0.044715 * cubic)
    return 0.5 * value * (1.0 + math.tanh(inner))


def attention(
    q: list[float],
    key_cache: list[list[float] | None],
    value_cache: list[list[float] | None],
    position: int,
) -> list[float]:
    """Compute causal multi-head attention for one token in the reference model."""
    head_dim = EMBEDDING_SIZE // HEAD_COUNT
    output = [0.0] * EMBEDDING_SIZE

    for head in range(HEAD_COUNT):
        base = head * head_dim
        scores: list[float] = []

        for time_index in range(position + 1):
            key = key_cache[time_index]
            if key is None:
                raise RuntimeError("Reference key cache is incomplete")

            dot = sum(
                q[base + d] * key[base + d]
                for d in range(head_dim)
            )
            scores.append(dot / math.sqrt(head_dim))

        maximum = max(scores)
        probabilities = [math.exp(score - maximum) for score in scores]
        normalizer = sum(probabilities)
        probabilities = [value / normalizer for value in probabilities]

        for d in range(head_dim):
            total = 0.0
            for time_index, probability in enumerate(probabilities):
                value = value_cache[time_index]
                if value is None:
                    raise RuntimeError("Reference value cache is incomplete")
                total += probability * value[base + d]
            output[base + d] = total

    return output


def forward(tokens: list[int]) -> list[float]:
    """Run an incremental GPT-2 reference forward pass for the supplied tokens."""
    key_cache = [
        [None] * CONTEXT_LENGTH
        for _ in range(LAYER_COUNT)
    ]
    value_cache = [
        [None] * CONTEXT_LENGTH
        for _ in range(LAYER_COUNT)
    ]

    token_embeddings = weights["token_embd.weight"][1]
    position_embeddings = weights["position_embd.weight"][1]
    logits: list[float] | None = None

    for position, token_id in enumerate(tokens):
        hidden = [
            token_embeddings[token_id * EMBEDDING_SIZE + i]
            + position_embeddings[position * EMBEDDING_SIZE + i]
            for i in range(EMBEDDING_SIZE)
        ]

        for layer in range(LAYER_COUNT):
            prefix = f"blk.{layer}."

            normalized = layer_norm(
                hidden,
                prefix + "attn_norm.weight",
                prefix + "attn_norm.bias",
            )
            qkv = add_bias(
                matvec(prefix + "attn_qkv.weight", normalized),
                prefix + "attn_qkv.bias",
            )
            q = qkv[:EMBEDDING_SIZE]
            k = qkv[EMBEDDING_SIZE : 2 * EMBEDDING_SIZE]
            v = qkv[2 * EMBEDDING_SIZE :]

            key_cache[layer][position] = k[:]
            value_cache[layer][position] = v[:]

            attended = attention(
                q,
                key_cache[layer],
                value_cache[layer],
                position,
            )
            projected = add_bias(
                matvec(prefix + "attn_output.weight", attended),
                prefix + "attn_output.bias",
            )
            hidden = [
                residual + update
                for residual, update in zip(hidden, projected)
            ]

            normalized = layer_norm(
                hidden,
                prefix + "ffn_norm.weight",
                prefix + "ffn_norm.bias",
            )
            feed_forward = add_bias(
                matvec(prefix + "ffn_up.weight", normalized),
                prefix + "ffn_up.bias",
            )
            feed_forward = [gelu(value) for value in feed_forward]
            down_projected = add_bias(
                matvec(prefix + "ffn_down.weight", feed_forward),
                prefix + "ffn_down.bias",
            )
            hidden = [
                residual + update
                for residual, update in zip(hidden, down_projected)
            ]

        normalized = layer_norm(
            hidden,
            "output_norm.weight",
            "output_norm.bias",
        )

        # GPT-2 ties the output projection to the token embedding matrix.
        logits = [
            sum(
                token_embeddings[token_id * EMBEDDING_SIZE + i] * normalized[i]
                for i in range(EMBEDDING_SIZE)
            )
            for token_id in range(VOCAB_SIZE)
        ]

    if logits is None:
        raise ValueError("The reference forward pass needs at least one token")
    return logits


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tiny-gpt2.gguf")
    output.parent.mkdir(parents=True, exist_ok=True)

    build_tensors()
    write_gguf(output)

    # The prompt "Hi" is byte-tokenized as IDs 72 and 105 in this fixture.
    reference_logits = forward([72, 105])
    expected_token = max(range(VOCAB_SIZE), key=reference_logits.__getitem__)

    expected_path = output.parent / "expected.txt"
    expected_path.write_text(f"{expected_token}\n", encoding="utf-8")

    print(f"wrote {output}; expected next token={expected_token}")


if __name__ == "__main__":
    main()
