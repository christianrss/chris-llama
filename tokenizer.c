#include "tokenizer.h"

#include <ctype.h>
#include <locale.h>
#include <wctype.h>

struct VocabSlot {
    const char *key;
    uint32_t id;
    bool used;
};

struct MergeSlot {
    char *left;
    char *right;
    uint32_t rank;
    bool used;
};

typedef struct {
    char **items;
    size_t count;
    size_t capacity;
} StringVector;

typedef struct {
    uint32_t *items;
    size_t count;
    size_t capacity;
} IdVector;

static uint64_t fnv1a_append(uint64_t hash,
                             const uint8_t *data,
                             size_t length) {
    for (size_t i = 0; i < length; ++i) {
        hash ^= data[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

static uint64_t hash_string(const char *text) {
    return fnv1a_append(1469598103934665603ull,
                        (const uint8_t *)text,
                        strlen(text));
}

static uint64_t hash_pair(const char *left, const char *right) {
    uint64_t hash = hash_string(left);
    const uint8_t separator = 0xff;
    hash = fnv1a_append(hash, &separator, 1);
    return fnv1a_append(hash,
                        (const uint8_t *)right,
                        strlen(right));
}

static size_t next_power_of_two(size_t value) {
    size_t result = 1;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

static void vocab_insert(GPT2Tokenizer *tokenizer,
                         const char *key,
                         uint32_t id) {
    size_t slot =
        (size_t)hash_string(key) & (tokenizer->vocab_cap - 1);

    while (tokenizer->vocab_table[slot].used) {
        slot = (slot + 1) & (tokenizer->vocab_cap - 1);
    }

    tokenizer->vocab_table[slot].used = true;
    tokenizer->vocab_table[slot].key = key;
    tokenizer->vocab_table[slot].id = id;
}

static int vocab_lookup(const GPT2Tokenizer *tokenizer,
                        const char *key,
                        uint32_t *id) {
    size_t slot =
        (size_t)hash_string(key) & (tokenizer->vocab_cap - 1);
    const size_t start = slot;

    while (tokenizer->vocab_table[slot].used) {
        if (strcmp(tokenizer->vocab_table[slot].key, key) == 0) {
            *id = tokenizer->vocab_table[slot].id;
            return 0;
        }

        slot = (slot + 1) & (tokenizer->vocab_cap - 1);
        if (slot == start) {
            break;
        }
    }

    return -1;
}

static void merge_insert(GPT2Tokenizer *tokenizer,
                         char *left,
                         char *right,
                         uint32_t rank) {
    size_t slot =
        (size_t)hash_pair(left, right) & (tokenizer->merge_cap - 1);

    while (tokenizer->merge_table[slot].used) {
        slot = (slot + 1) & (tokenizer->merge_cap - 1);
    }

    tokenizer->merge_table[slot].used = true;
    tokenizer->merge_table[slot].left = left;
    tokenizer->merge_table[slot].right = right;
    tokenizer->merge_table[slot].rank = rank;
}

static int merge_rank(const GPT2Tokenizer *tokenizer,
                      const char *left,
                      const char *right,
                      uint32_t *rank) {
    size_t slot =
        (size_t)hash_pair(left, right) & (tokenizer->merge_cap - 1);
    const size_t start = slot;

    while (tokenizer->merge_table[slot].used) {
        const MergeSlot *entry = &tokenizer->merge_table[slot];
        if (strcmp(entry->left, left) == 0 &&
            strcmp(entry->right, right) == 0) {
            *rank = entry->rank;
            return 0;
        }

        slot = (slot + 1) & (tokenizer->merge_cap - 1);
        if (slot == start) {
            break;
        }
    }

    return -1;
}

static size_t utf8_encode(uint32_t codepoint, char out[5]) {
    if (codepoint <= 0x7f) {
        out[0] = (char)codepoint;
        out[1] = '\0';
        return 1;
    }

    if (codepoint <= 0x7ff) {
        out[0] = (char)(0xc0 | (codepoint >> 6));
        out[1] = (char)(0x80 | (codepoint & 0x3f));
        out[2] = '\0';
        return 2;
    }

    if (codepoint <= 0xffff) {
        out[0] = (char)(0xe0 | (codepoint >> 12));
        out[1] = (char)(0x80 | ((codepoint >> 6) & 0x3f));
        out[2] = (char)(0x80 | (codepoint & 0x3f));
        out[3] = '\0';
        return 3;
    }

    out[0] = (char)(0xf0 | (codepoint >> 18));
    out[1] = (char)(0x80 | ((codepoint >> 12) & 0x3f));
    out[2] = (char)(0x80 | ((codepoint >> 6) & 0x3f));
    out[3] = (char)(0x80 | (codepoint & 0x3f));
    out[4] = '\0';
    return 4;
}

static uint32_t utf8_decode_one(const uint8_t *data,
                                size_t length,
                                size_t *used) {
    if (!length) {
        *used = 0;
        return 0;
    }

    const uint8_t first = data[0];
    if (first < 0x80) {
        *used = 1;
        return first;
    }

    if ((first & 0xe0) == 0xc0 &&
        length >= 2 &&
        (data[1] & 0xc0) == 0x80) {
        *used = 2;
        return ((uint32_t)(first & 0x1f) << 6) |
               (data[1] & 0x3f);
    }

    if ((first & 0xf0) == 0xe0 &&
        length >= 3 &&
        (data[1] & 0xc0) == 0x80 &&
        (data[2] & 0xc0) == 0x80) {
        *used = 3;
        return ((uint32_t)(first & 0x0f) << 12) |
               ((uint32_t)(data[1] & 0x3f) << 6) |
               (data[2] & 0x3f);
    }

    if ((first & 0xf8) == 0xf0 &&
        length >= 4 &&
        (data[1] & 0xc0) == 0x80 &&
        (data[2] & 0xc0) == 0x80 &&
        (data[3] & 0xc0) == 0x80) {
        *used = 4;
        return ((uint32_t)(first & 0x07) << 18) |
               ((uint32_t)(data[1] & 0x3f) << 12) |
               ((uint32_t)(data[2] & 0x3f) << 6) |
               (data[3] & 0x3f);
    }

    /* Keep invalid UTF-8 bytes lossless instead of dropping them. */
    *used = 1;
    return first;
}

/*
 * GPT-2 maps bytes to a reversible Unicode alphabet before applying BPE.
 * Rebuild that mapping here so the tokenizer can be reconstructed from GGUF
 * metadata alone.
 */
static void build_byte_unicode(uint32_t codepoint_for_byte[256],
                               int byte_for_codepoint[512]) {
    bool direct[256] = {false};

    for (int byte = 33; byte <= 126; ++byte) {
        direct[byte] = true;
    }
    for (int byte = 161; byte <= 172; ++byte) {
        direct[byte] = true;
    }
    for (int byte = 174; byte <= 255; ++byte) {
        direct[byte] = true;
    }

    for (int i = 0; i < 512; ++i) {
        byte_for_codepoint[i] = -1;
    }

    uint32_t extra = 0;
    for (int byte = 0; byte < 256; ++byte) {
        const uint32_t codepoint = direct[byte]
                                       ? (uint32_t)byte
                                       : 256u + extra++;
        codepoint_for_byte[byte] = codepoint;

        if (codepoint < 512) {
            byte_for_codepoint[codepoint] = byte;
        }
    }
}

int tokenizer_init_from_gguf(GPT2Tokenizer *tokenizer,
                             const GGUFFile *gguf) {
    memset(tokenizer, 0, sizeof(*tokenizer));

    const GGUFArray *tokens = gguf_get_array(
        gguf,
        "tokenizer.ggml.tokens",
        GGUF_VALUE_STRING);
    const GGUFArray *merges = gguf_get_array(
        gguf,
        "tokenizer.ggml.merges",
        GGUF_VALUE_STRING);

    if (!tokens || tokens->count == 0) {
        fprintf(stderr,
                "error: GGUF does not contain tokenizer.ggml.tokens\n");
        return -1;
    }

    tokenizer->n_tokens = tokens->count;
    tokenizer->tokens = (char **)chris_xcalloc(
        (size_t)tokenizer->n_tokens,
        sizeof(char *));

    char **source_tokens = (char **)tokens->data;
    for (uint64_t i = 0; i < tokenizer->n_tokens; ++i) {
        tokenizer->tokens[i] = chris_xstrdup(source_tokens[i]);
    }

    const GGUFArray *types = gguf_get_array(
        gguf,
        "tokenizer.ggml.token_type",
        GGUF_VALUE_INT32);

    if (types && types->count == tokenizer->n_tokens) {
        tokenizer->token_types = (int32_t *)chris_xmalloc(
            (size_t)tokenizer->n_tokens * sizeof(int32_t));
        memcpy(tokenizer->token_types,
               types->data,
               (size_t)tokenizer->n_tokens * sizeof(int32_t));
    }

    tokenizer->n_merges = merges ? merges->count : 0;
    tokenizer->merges = (char **)chris_xcalloc(
        (size_t)(tokenizer->n_merges ? tokenizer->n_merges : 1),
        sizeof(char *));

    if (merges) {
        char **source_merges = (char **)merges->data;
        for (uint64_t i = 0; i < tokenizer->n_merges; ++i) {
            tokenizer->merges[i] = chris_xstrdup(source_merges[i]);
        }
    }

    tokenizer->vocab_cap = next_power_of_two(
        (size_t)tokenizer->n_tokens * 2 + 1);
    tokenizer->vocab_table = (VocabSlot *)chris_xcalloc(
        tokenizer->vocab_cap,
        sizeof(VocabSlot));

    for (uint64_t i = 0; i < tokenizer->n_tokens; ++i) {
        vocab_insert(tokenizer, tokenizer->tokens[i], (uint32_t)i);
    }

    tokenizer->merge_cap = next_power_of_two(
        (size_t)tokenizer->n_merges * 2 + 1);
    tokenizer->merge_table = (MergeSlot *)chris_xcalloc(
        tokenizer->merge_cap,
        sizeof(MergeSlot));

    for (uint64_t i = 0; i < tokenizer->n_merges; ++i) {
        char *line = tokenizer->merges[i];
        char *separator = strchr(line, ' ');
        if (!separator) {
            continue;
        }

        /*
         * Split the copied merge string in place. Both pointers remain valid
         * until tokenizer_free() releases the original allocation.
         */
        *separator = '\0';
        merge_insert(tokenizer,
                     line,
                     separator + 1,
                     (uint32_t)i);
    }

    uint64_t id = 0;
    if (gguf_get_u64(gguf, "tokenizer.ggml.bos_token_id", &id) == 0 &&
        id < tokenizer->n_tokens) {
        tokenizer->bos_id = (uint32_t)id;
        tokenizer->has_bos = true;
    }

    if (gguf_get_u64(gguf, "tokenizer.ggml.eos_token_id", &id) == 0 &&
        id < tokenizer->n_tokens) {
        tokenizer->eos_id = (uint32_t)id;
        tokenizer->has_eos = true;
    }

    if (!tokenizer->has_eos) {
        uint32_t end_of_text = 0;
        if (vocab_lookup(tokenizer,
                         "<|endoftext|>",
                         &end_of_text) == 0) {
            tokenizer->eos_id = end_of_text;
            tokenizer->has_eos = true;
        }
    }

    uint32_t unused_forward_map[256];
    build_byte_unicode(unused_forward_map, tokenizer->byte_for_cp);

    /* iswalpha/iswspace use the process locale for non-ASCII codepoints. */
    setlocale(LC_CTYPE, "");
    return 0;
}

void tokenizer_free(GPT2Tokenizer *tokenizer) {
    if (!tokenizer) {
        return;
    }

    for (uint64_t i = 0; i < tokenizer->n_tokens; ++i) {
        free(tokenizer->tokens ? tokenizer->tokens[i] : NULL);
    }

    for (uint64_t i = 0; i < tokenizer->n_merges; ++i) {
        free(tokenizer->merges ? tokenizer->merges[i] : NULL);
    }

    free(tokenizer->tokens);
    free(tokenizer->merges);
    free(tokenizer->token_types);
    free(tokenizer->vocab_table);
    free(tokenizer->merge_table);

    memset(tokenizer, 0, sizeof(*tokenizer));
}

static void string_vector_push(StringVector *vector, char *value) {
    if (vector->count == vector->capacity) {
        const size_t new_capacity = vector->capacity
                                        ? vector->capacity * 2
                                        : 16;
        char **new_items = (char **)realloc(
            vector->items,
            new_capacity * sizeof(char *));
        if (!new_items) {
            fprintf(stderr, "error: tokenizer allocation failed\n");
            exit(1);
        }
        vector->items = new_items;
        vector->capacity = new_capacity;
    }

    vector->items[vector->count++] = value;
}

static void id_vector_push(IdVector *vector, uint32_t id) {
    if (vector->count == vector->capacity) {
        const size_t new_capacity = vector->capacity
                                        ? vector->capacity * 2
                                        : 32;
        uint32_t *new_items = (uint32_t *)realloc(
            vector->items,
            new_capacity * sizeof(uint32_t));
        if (!new_items) {
            fprintf(stderr, "error: tokenizer allocation failed\n");
            exit(1);
        }
        vector->items = new_items;
        vector->capacity = new_capacity;
    }

    vector->items[vector->count++] = id;
}

static char *concat_symbols(const char *left, const char *right) {
    const size_t left_length = strlen(left);
    const size_t right_length = strlen(right);
    char *joined = (char *)chris_xmalloc(
        left_length + right_length + 1);

    memcpy(joined, left, left_length);
    memcpy(joined + left_length,
           right,
           right_length + 1);
    return joined;
}

static int bpe_piece(const GPT2Tokenizer *tokenizer,
                     const uint8_t *raw,
                     size_t length,
                     IdVector *output) {
    uint32_t codepoint_for_byte[256];
    int unused_inverse[512];
    build_byte_unicode(codepoint_for_byte, unused_inverse);

    StringVector symbols = {0};

    for (size_t i = 0; i < length; ++i) {
        char encoded[5];
        utf8_encode(codepoint_for_byte[raw[i]], encoded);
        string_vector_push(&symbols, chris_xstrdup(encoded));
    }

    if (!symbols.count) {
        free(symbols.items);
        return 0;
    }

    while (symbols.count > 1) {
        uint32_t best_rank = UINT32_MAX;
        bool found = false;

        for (size_t i = 0; i + 1 < symbols.count; ++i) {
            uint32_t rank = 0;
            if (merge_rank(tokenizer,
                           symbols.items[i],
                           symbols.items[i + 1],
                           &rank) == 0 &&
                (!found || rank < best_rank)) {
                best_rank = rank;
                found = true;
            }
        }

        if (!found) {
            break;
        }

        StringVector next = {0};
        for (size_t i = 0; i < symbols.count;) {
            uint32_t rank = 0;
            const bool can_merge =
                i + 1 < symbols.count &&
                merge_rank(tokenizer,
                           symbols.items[i],
                           symbols.items[i + 1],
                           &rank) == 0 &&
                rank == best_rank;

            if (can_merge) {
                char *merged = concat_symbols(symbols.items[i],
                                              symbols.items[i + 1]);
                free(symbols.items[i]);
                free(symbols.items[i + 1]);
                string_vector_push(&next, merged);
                i += 2;
            } else {
                string_vector_push(&next, symbols.items[i]);
                ++i;
            }
        }

        free(symbols.items);
        symbols = next;
    }

    for (size_t i = 0; i < symbols.count; ++i) {
        uint32_t id = 0;
        if (vocab_lookup(tokenizer, symbols.items[i], &id) != 0) {
            fprintf(stderr,
                    "error: tokenizer: BPE symbol is missing from vocabulary: "
                    "'%s'\n",
                    symbols.items[i]);

            for (size_t j = i; j < symbols.count; ++j) {
                free(symbols.items[j]);
            }
            free(symbols.items);
            return -1;
        }

        id_vector_push(output, id);
        free(symbols.items[i]);
    }

    free(symbols.items);
    return 0;
}

typedef enum {
    CHAR_SPACE,
    CHAR_LETTER,
    CHAR_DIGIT,
    CHAR_OTHER,
} CharClass;

static CharClass classify_character(const uint8_t *data,
                                    size_t length,
                                    size_t *used) {
    const uint32_t codepoint = utf8_decode_one(data, length, used);

    if (codepoint < 128) {
        if (isspace((unsigned char)codepoint)) {
            return CHAR_SPACE;
        }
        if (isalpha((unsigned char)codepoint)) {
            return CHAR_LETTER;
        }
        if (isdigit((unsigned char)codepoint)) {
            return CHAR_DIGIT;
        }
        return CHAR_OTHER;
    }

    const wint_t wide = (wint_t)codepoint;
    if (iswspace(wide)) {
        return CHAR_SPACE;
    }
    if (iswdigit(wide)) {
        return CHAR_DIGIT;
    }
    if (iswalpha(wide)) {
        return CHAR_LETTER;
    }
    return CHAR_OTHER;
}

static size_t contraction_length(const uint8_t *data, size_t length) {
    static const char *contractions[] = {
        "'s", "'t", "'re", "'ve", "'m", "'ll", "'d",
    };

    for (size_t i = 0;
         i < sizeof(contractions) / sizeof(contractions[0]);
         ++i) {
        const size_t candidate_length = strlen(contractions[i]);
        if (length >= candidate_length &&
            memcmp(data, contractions[i], candidate_length) == 0) {
            return candidate_length;
        }
    }

    return 0;
}

/*
 * Reproduce the useful boundaries from the GPT-2 pre-tokenizer regex:
 * contractions, optional leading spaces, runs of letters/numbers/symbols, and
 * whitespace groups. BPE itself is applied by bpe_piece().
 */
static int pretokenize_and_bpe(const GPT2Tokenizer *tokenizer,
                               const uint8_t *data,
                               size_t length,
                               IdVector *output) {
    size_t offset = 0;

    while (offset < length) {
        const size_t contraction =
            contraction_length(data + offset, length - offset);
        if (contraction) {
            if (bpe_piece(tokenizer,
                          data + offset,
                          contraction,
                          output)) {
                return -1;
            }
            offset += contraction;
            continue;
        }

        size_t used = 0;
        const CharClass current =
            classify_character(data + offset,
                               length - offset,
                               &used);

        /* GPT-2 joins one ordinary ASCII space to the following token class. */
        if (current == CHAR_SPACE &&
            data[offset] == ' ' &&
            offset + 1 < length) {
            size_t next_used = 0;
            const CharClass next_class =
                classify_character(data + offset + 1,
                                   length - offset - 1,
                                   &next_used);

            if (next_class != CHAR_SPACE) {
                size_t end = offset + 1;
                while (end < length) {
                    size_t char_size = 0;
                    const CharClass candidate =
                        classify_character(data + end,
                                           length - end,
                                           &char_size);
                    if (candidate != next_class) {
                        break;
                    }
                    end += char_size;
                }

                if (bpe_piece(tokenizer,
                              data + offset,
                              end - offset,
                              output)) {
                    return -1;
                }
                offset = end;
                continue;
            }
        }

        if (current == CHAR_SPACE) {
            size_t end = offset;
            size_t last_whitespace = offset;
            size_t whitespace_count = 0;

            while (end < length) {
                size_t char_size = 0;
                if (classify_character(data + end,
                                       length - end,
                                       &char_size) != CHAR_SPACE) {
                    break;
                }

                last_whitespace = end;
                end += char_size;
                ++whitespace_count;
            }

            /*
             * When whitespace is followed by text, GPT-2 leaves the final
             * whitespace codepoint for the next pre-tokenized piece.
             */
            if (end < length && whitespace_count > 1) {
                end = last_whitespace;
            }

            if (bpe_piece(tokenizer,
                          data + offset,
                          end - offset,
                          output)) {
                return -1;
            }
            offset = end;
            continue;
        }

        size_t end = offset;
        while (end < length) {
            if (end > offset &&
                contraction_length(data + end, length - end)) {
                break;
            }

            size_t char_size = 0;
            const CharClass candidate =
                classify_character(data + end,
                                   length - end,
                                   &char_size);
            if (candidate != current) {
                break;
            }
            end += char_size;
        }

        if (bpe_piece(tokenizer,
                      data + offset,
                      end - offset,
                      output)) {
            return -1;
        }
        offset = end;
    }

    return 0;
}

int tokenizer_encode(const GPT2Tokenizer *tokenizer,
                     const char *text,
                     uint32_t **ids,
                     size_t *count) {
    if (!tokenizer || !text || !ids || !count) {
        return -1;
    }

    IdVector output = {0};
    if (pretokenize_and_bpe(tokenizer,
                            (const uint8_t *)text,
                            strlen(text),
                            &output) != 0) {
        free(output.items);
        return -1;
    }

    *ids = output.items;
    *count = output.count;
    return 0;
}

size_t tokenizer_decode_token(const GPT2Tokenizer *tokenizer,
                              uint32_t id,
                              uint8_t *out,
                              size_t capacity) {
    if (!tokenizer ||
        id >= tokenizer->n_tokens ||
        !out ||
        !capacity) {
        return 0;
    }

    const uint8_t *encoded = (const uint8_t *)tokenizer->tokens[id];
    const size_t encoded_length = strlen(tokenizer->tokens[id]);
    size_t offset = 0;
    size_t written = 0;

    while (offset < encoded_length) {
        size_t used = 0;
        const uint32_t codepoint =
            utf8_decode_one(encoded + offset,
                            encoded_length - offset,
                            &used);
        if (!used) {
            break;
        }

        if (codepoint < 512 && tokenizer->byte_for_cp[codepoint] >= 0) {
            if (written < capacity) {
                out[written] =
                    (uint8_t)tokenizer->byte_for_cp[codepoint];
            }
            ++written;
        } else {
            /* Special tokens bypass the byte/Unicode map. */
            for (size_t i = 0; i < used; ++i) {
                if (written < capacity) {
                    out[written] = encoded[offset + i];
                }
                ++written;
            }
        }

        offset += used;
    }

    return written;
}

void tokenizer_print_token(const GPT2Tokenizer *tokenizer,
                           uint32_t id,
                           FILE *stream) {
    uint8_t buffer[4096];
    size_t length = tokenizer_decode_token(tokenizer,
                                           id,
                                           buffer,
                                           sizeof(buffer));
    if (length > sizeof(buffer)) {
        length = sizeof(buffer);
    }

    fwrite(buffer, 1, length, stream);
    fflush(stream);
}
