#ifndef CHRIS_TOKENIZER_H
#define CHRIS_TOKENIZER_H

#include "gguf.h"

#include <stdio.h>

typedef struct VocabSlot VocabSlot;
typedef struct MergeSlot MergeSlot;

typedef struct {
    char **tokens;       /* Strings as stored in GGUF metadata. */
    uint64_t n_tokens;

    char **merges;
    uint64_t n_merges;

    uint32_t bos_id;
    uint32_t eos_id;
    bool has_bos;
    bool has_eos;

    int32_t *token_types;  /* Optional tokenizer.ggml.token_type metadata. */

    VocabSlot *vocab_table;
    size_t vocab_cap;
    MergeSlot *merge_table;
    size_t merge_cap;

    int byte_for_cp[512];  /* Inverse GPT-2 bytes_to_unicode table. */
} GPT2Tokenizer;

int tokenizer_init_from_gguf(GPT2Tokenizer *tokenizer,
                             const GGUFFile *gguf);
void tokenizer_free(GPT2Tokenizer *tokenizer);

/* Allocates *ids. The caller owns the returned array. */
int tokenizer_encode(const GPT2Tokenizer *tokenizer,
                     const char *text,
                     uint32_t **ids,
                     size_t *count);

size_t tokenizer_decode_token(const GPT2Tokenizer *tokenizer,
                              uint32_t id,
                              uint8_t *out,
                              size_t capacity);
void tokenizer_print_token(const GPT2Tokenizer *tokenizer,
                           uint32_t id,
                           FILE *stream);

#endif /* CHRIS_TOKENIZER_H */
