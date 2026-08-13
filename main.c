#include "gguf.h"
#include "gpt2.h"
#include "sampler.h"
#include "tokenizer.h"

#include <time.h>

typedef struct {
    const char *model_path;
    const char *prompt;
    const char *tokenize_text;
    uint32_t max_tokens;
    const char *device_spec;
    bool inspect;
    bool list_tensors;
    bool interactive;
    bool next_token_id;
    bool list_devices;
    Sampler sampler;
} Options;

static void usage(const char *argv0) {
    printf("Chris Llama / Chris-GPT GGUF runtime\n\n");
    printf("Usage:\n");
    printf("  %s MODEL.gguf -p \"prompt\" [options]\n", argv0);
    printf("  %s MODEL.gguf --interactive [options]\n", argv0);
    printf("  %s MODEL.gguf --inspect [--list-tensors]\n", argv0);
    printf("  %s MODEL.gguf --tokenize \"text\"\n", argv0);
    printf("  %s --list-devices\n\n", argv0);

    printf("Options:\n");
    printf("  -p, --prompt TEXT          Prompt to complete\n");
    printf("  -n, --max-tokens N        Number of new tokens (default: 64)\n");
    printf("  --temperature X           0 = greedy (default: 0.8)\n");
    printf("  --top-k N                 Top-k cutoff (default: 40; 0 disables it)\n");
    printf("  --top-p X                 Nucleus sampling threshold (default: 0.95)\n");
    printf("  --repeat-penalty X        Repetition penalty (default: 1.0)\n");
    printf("  --repeat-last-n N         Repetition window (default: 64)\n");
    printf("  --seed N                  Sampler seed\n");
    printf("  --device SPEC             auto|gpu|cpu|N|substring (default: auto)\n");
    printf("  --list-devices            Print devices exposed by the selected backend\n");
    printf("  -i, --interactive         Read one completion prompt per line\n");
    printf("  --inspect                 Print GGUF and GPT-2 metadata\n");
    printf("  --list-tensors            Print tensor names, shapes, and GGML types\n");
    printf("  --tokenize TEXT           Print GPT-2 BPE token IDs\n");
    printf("  --next-token-id           Print only the next greedy token ID\n");
    printf("  -h, --help                Show this help\n");
}

static int parse_u32(const char *text, uint32_t *out) {
    char *end = NULL;
    errno = 0;

    const unsigned long value = strtoul(text, &end, 10);
    if (errno || !end || *end != '\0' || value > UINT32_MAX) {
        return -1;
    }

    *out = (uint32_t)value;
    return 0;
}

static int parse_float(const char *text, float *out) {
    char *end = NULL;
    errno = 0;

    const float value = strtof(text, &end);
    if (errno || !end || *end != '\0' || !isfinite(value)) {
        return -1;
    }

    *out = value;
    return 0;
}

static int parse_u64(const char *text, uint64_t *out) {
    char *end = NULL;
    errno = 0;

    const unsigned long long value = strtoull(text, &end, 10);
    if (errno || !end || *end != '\0') {
        return -1;
    }

    *out = (uint64_t)value;
    return 0;
}

static int parse_options(int argc, char **argv, Options *options) {
    memset(options, 0, sizeof(*options));
    options->max_tokens = 64;
    options->device_spec = "auto";
    sampler_init(&options->sampler, (uint64_t)time(NULL));

    if (argc < 2) {
        return -1;
    }

    if (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
        return 1;
    }

    if (!strcmp(argv[1], "--list-devices")) {
        options->list_devices = true;
        return 0;
    }

    options->model_path = argv[1];

    for (int i = 2; i < argc; ++i) {
#define REQUIRE_VALUE()                                                        \
    do {                                                                       \
        if (i + 1 >= argc) {                                                   \
            fprintf(stderr, "error: %s requires a value\n", argv[i]);        \
            return -1;                                                         \
        }                                                                      \
    } while (0)

        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            return 1;
        } else if (!strcmp(argv[i], "-p") || !strcmp(argv[i], "--prompt")) {
            REQUIRE_VALUE();
            options->prompt = argv[++i];
        } else if (!strcmp(argv[i], "-n") ||
                   !strcmp(argv[i], "--max-tokens")) {
            REQUIRE_VALUE();
            if (parse_u32(argv[++i], &options->max_tokens)) {
                return -1;
            }
        } else if (!strcmp(argv[i], "--device")) {
            REQUIRE_VALUE();
            options->device_spec = argv[++i];
        } else if (!strcmp(argv[i], "--list-devices")) {
            options->list_devices = true;
        } else if (!strcmp(argv[i], "--temperature")) {
            REQUIRE_VALUE();
            if (parse_float(argv[++i], &options->sampler.temperature)) {
                return -1;
            }
        } else if (!strcmp(argv[i], "--top-k")) {
            REQUIRE_VALUE();
            if (parse_u32(argv[++i], &options->sampler.top_k)) {
                return -1;
            }
        } else if (!strcmp(argv[i], "--top-p")) {
            REQUIRE_VALUE();
            if (parse_float(argv[++i], &options->sampler.top_p)) {
                return -1;
            }
        } else if (!strcmp(argv[i], "--repeat-penalty")) {
            REQUIRE_VALUE();
            if (parse_float(argv[++i], &options->sampler.repeat_penalty)) {
                return -1;
            }
        } else if (!strcmp(argv[i], "--repeat-last-n")) {
            REQUIRE_VALUE();
            if (parse_u32(argv[++i], &options->sampler.repeat_last_n)) {
                return -1;
            }
        } else if (!strcmp(argv[i], "--seed")) {
            REQUIRE_VALUE();
            if (parse_u64(argv[++i], &options->sampler.rng_state)) {
                return -1;
            }
        } else if (!strcmp(argv[i], "-i") ||
                   !strcmp(argv[i], "--interactive")) {
            options->interactive = true;
        } else if (!strcmp(argv[i], "--inspect")) {
            options->inspect = true;
        } else if (!strcmp(argv[i], "--list-tensors")) {
            options->inspect = true;
            options->list_tensors = true;
        } else if (!strcmp(argv[i], "--tokenize")) {
            REQUIRE_VALUE();
            options->tokenize_text = argv[++i];
        } else if (!strcmp(argv[i], "--next-token-id")) {
            options->next_token_id = true;
        } else {
            fprintf(stderr, "error: unknown option: %s\n", argv[i]);
            return -1;
        }

#undef REQUIRE_VALUE
    }

    if (options->sampler.top_p < 0.0f ||
        options->sampler.top_p > 1.0f ||
        options->sampler.repeat_penalty < 1.0f) {
        return -1;
    }

    return 0;
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static int append_id(uint32_t **items,
                     size_t *count,
                     size_t *capacity,
                     uint32_t id) {
    if (*count == *capacity) {
        const size_t new_capacity = *capacity ? *capacity * 2 : 128;
        uint32_t *new_items =
            (uint32_t *)realloc(*items, new_capacity * sizeof(uint32_t));
        if (!new_items) {
            return -1;
        }

        *items = new_items;
        *capacity = new_capacity;
    }

    (*items)[(*count)++] = id;
    return 0;
}

static void mask_unused_tokens(const GPT2Tokenizer *tokenizer,
                               float *logits,
                               uint32_t vocab_size) {
    /*
     * GGUF token type 5 is UNUSED. Some converters pad the embedding matrix
     * with placeholder vocabulary entries; those IDs must never be sampled.
     */
    if (!tokenizer->token_types) {
        return;
    }

    for (uint32_t id = 0; id < vocab_size; ++id) {
        const bool is_eos = tokenizer->has_eos && id == tokenizer->eos_id;
        if (tokenizer->token_types[id] == 5 && !is_eos) {
            logits[id] = -INFINITY;
        }
    }
}

static int run_prompt(GPT2Model *model,
                      const GPT2Tokenizer *tokenizer,
                      Options *options,
                      const char *prompt,
                      bool decorate) {
    uint32_t *prompt_ids = NULL;
    size_t prompt_count = 0;

    if (tokenizer_encode(tokenizer, prompt, &prompt_ids, &prompt_count) != 0) {
        return -1;
    }

    if (prompt_count == 0) {
        if (!tokenizer->has_eos) {
            fprintf(stderr,
                    "error: empty prompt and tokenizer has no BOS/EOS token\n");
            return -1;
        }

        prompt_ids = (uint32_t *)chris_xmalloc(sizeof(uint32_t));
        prompt_ids[0] = tokenizer->eos_id;
        prompt_count = 1;
    }

    const uint32_t context_length = model->cfg.context_length;
    const size_t max_prompt = context_length > 1 ? context_length - 1 : 1;

    if (prompt_count > max_prompt) {
        fprintf(stderr,
                "warning: prompt has %zu tokens; keeping the last %zu\n",
                prompt_count,
                max_prompt);
        memmove(prompt_ids,
                prompt_ids + prompt_count - max_prompt,
                max_prompt * sizeof(uint32_t));
        prompt_count = max_prompt;
    }

    for (size_t i = 0; i < prompt_count; ++i) {
        if (prompt_ids[i] >= model->cfg.vocab_size) {
            fprintf(stderr,
                    "error: token %u is outside model vocab_size=%u\n",
                    prompt_ids[i],
                    model->cfg.vocab_size);
            free(prompt_ids);
            return -1;
        }
    }

    gpt2_reset(model);

    float *logits = (float *)chris_xmalloc(
        (size_t)model->cfg.vocab_size * sizeof(float));

    const double prompt_start = now_seconds();
    for (size_t i = 0; i < prompt_count; ++i) {
        float *token_logits = (i + 1 == prompt_count) ? logits : NULL;
        if (gpt2_forward_token(model,
                               prompt_ids[i],
                               (uint32_t)i,
                               token_logits) != 0) {
            free(prompt_ids);
            free(logits);
            return -1;
        }
    }
    const double prompt_end = now_seconds();

    const uint32_t sampling_vocab = (uint32_t)CHRIS_MIN(
        (uint64_t)model->cfg.vocab_size,
        tokenizer->n_tokens);

    if (options->next_token_id) {
        Sampler greedy = options->sampler;
        greedy.temperature = 0.0f;
        mask_unused_tokens(tokenizer, logits, sampling_vocab);

        const uint32_t id = sampler_sample(
            &greedy,
            logits,
            sampling_vocab,
            prompt_ids,
            prompt_count);

        printf("%u\n", id);
        free(prompt_ids);
        free(logits);
        return 0;
    }

    if (decorate) {
        printf("Model> ");
    } else {
        fwrite(prompt, 1, strlen(prompt), stdout);
        fflush(stdout);
    }

    size_t history_capacity = prompt_count + options->max_tokens + 8;
    uint32_t *history = (uint32_t *)chris_xmalloc(
        history_capacity * sizeof(uint32_t));
    memcpy(history, prompt_ids, prompt_count * sizeof(uint32_t));

    size_t history_count = prompt_count;
    uint32_t position = (uint32_t)prompt_count;
    uint32_t generated = 0;
    const double generation_start = now_seconds();

    while (generated < options->max_tokens && position < context_length) {
        mask_unused_tokens(tokenizer, logits, sampling_vocab);

        const uint32_t next = sampler_sample(
            &options->sampler,
            logits,
            sampling_vocab,
            history,
            history_count);

        if (tokenizer->has_eos && next == tokenizer->eos_id) {
            break;
        }

        tokenizer_print_token(tokenizer, next, stdout);

        if (append_id(&history,
                      &history_count,
                      &history_capacity,
                      next) != 0) {
            free(history);
            free(prompt_ids);
            free(logits);
            return -1;
        }

        ++generated;
        if (gpt2_forward_token(model, next, position, logits) != 0) {
            free(history);
            free(prompt_ids);
            free(logits);
            return -1;
        }
        ++position;
    }

    const double generation_end = now_seconds();
    const double prompt_seconds = prompt_end - prompt_start;
    const double generation_seconds = generation_end - generation_start;

    printf("\n");
    fprintf(stderr,
            "[timing] prompt: %zu tokens in %.3fs (%.2f tok/s) | "
            "generation: %u tokens in %.3fs (%.2f tok/s)\n",
            prompt_count,
            prompt_seconds,
            prompt_count / (prompt_seconds + 1e-12),
            generated,
            generation_seconds,
            generated / (generation_seconds + 1e-12));

    free(history);
    free(prompt_ids);
    free(logits);
    return 0;
}

static int run_interactive(GPT2Model *model,
                           const GPT2Tokenizer *tokenizer,
                           Options *options) {
    char line[8192];

    printf("Chris-GPT completion runtime. Type 'exit' to quit.\n");

    while (true) {
        printf("Prompt> ");
        fflush(stdout);

        if (!fgets(line, sizeof(line), stdin)) {
            break;
        }

        size_t length = strlen(line);
        if (length && line[length - 1] == '\n') {
            line[--length] = '\0';
        }

        if (!strcmp(line, "exit") || !strcmp(line, "quit")) {
            break;
        }

        if (run_prompt(model, tokenizer, options, line, true) != 0) {
            return -1;
        }
    }

    return 0;
}

int main(int argc, char **argv) {
    Options options;
    const int parse_result = parse_options(argc, argv, &options);

    if (parse_result != 0) {
        usage(argv[0]);
        return parse_result > 0 ? 0 : 1;
    }

    if (options.list_devices) {
        chris_backend_list_devices();
        return 0;
    }

    GGUFFile gguf;
    if (gguf_open(options.model_path, &gguf) != 0) {
        return 1;
    }

    if (options.inspect) {
        gguf_print_summary(&gguf, options.list_tensors);

        if (!options.prompt &&
            !options.interactive &&
            !options.tokenize_text &&
            !options.next_token_id) {
            gguf_close(&gguf);
            return 0;
        }
    }

    GPT2Tokenizer tokenizer;
    if (tokenizer_init_from_gguf(&tokenizer, &gguf) != 0) {
        gguf_close(&gguf);
        return 1;
    }

    if (options.tokenize_text) {
        uint32_t *ids = NULL;
        size_t count = 0;

        if (tokenizer_encode(&tokenizer,
                             options.tokenize_text,
                             &ids,
                             &count) != 0) {
            tokenizer_free(&tokenizer);
            gguf_close(&gguf);
            return 1;
        }

        printf("tokens[%zu] =", count);
        for (size_t i = 0; i < count; ++i) {
            printf(" %u", ids[i]);
        }
        printf("\n");
        free(ids);

        if (!options.prompt &&
            !options.interactive &&
            !options.next_token_id) {
            tokenizer_free(&tokenizer);
            gguf_close(&gguf);
            return 0;
        }
    }

    GPT2Model model;
    if (gpt2_load(&model, &gguf) != 0) {
        tokenizer_free(&tokenizer);
        gguf_close(&gguf);
        return 1;
    }

    gpt2_print_config(&model);
    fprintf(stderr,
            "[tokenizer] %" PRIu64 " tokens, %" PRIu64
            " merges; sampling limit=%" PRIu64 "\n",
            tokenizer.n_tokens,
            tokenizer.n_merges,
            CHRIS_MIN((uint64_t)model.cfg.vocab_size, tokenizer.n_tokens));

    ChrisBackend *backend = chris_backend_create(options.device_spec);
    if (!backend) {
        gpt2_free(&model);
        tokenizer_free(&tokenizer);
        gguf_close(&gguf);
        return 1;
    }

    fprintf(stderr,
            "[runtime] backend=%s device=%s accelerated=%s\n",
            chris_backend_name(backend),
            chris_backend_device_name(backend),
            chris_backend_is_accelerated(backend) ? "yes" : "no");

    if (gpt2_attach_backend(&model, backend) != 0) {
        chris_backend_destroy(backend);
        gpt2_free(&model);
        tokenizer_free(&tokenizer);
        gguf_close(&gguf);
        return 1;
    }

    int result = 0;
    if (options.interactive) {
        result = run_interactive(&model, &tokenizer, &options);
    } else if (options.prompt) {
        result = run_prompt(&model,
                            &tokenizer,
                            &options,
                            options.prompt,
                            false);
    } else {
        fprintf(stderr, "error: provide -p PROMPT or --interactive\n");
        result = -1;
    }

    gpt2_free(&model);
    chris_backend_destroy(backend);
    tokenizer_free(&tokenizer);
    gguf_close(&gguf);

    return result == 0 ? 0 : 1;
}
