#include "sampler.h"

typedef struct {
    uint32_t id;
    float logit;
    float probability;
} Candidate;

static uint64_t rng_next(uint64_t *state) {
    uint64_t x = *state;
    if (!x) {
        x = 0x9e3779b97f4a7c15ull;
    }

    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 2685821657736338717ull;
}

static double rng_unit(uint64_t *state) {
    return (rng_next(state) >> 11) *
           (1.0 / 9007199254740992.0);
}

void sampler_init(Sampler *sampler, uint64_t seed) {
    memset(sampler, 0, sizeof(*sampler));
    sampler->temperature = 0.8f;
    sampler->top_k = 40;
    sampler->top_p = 0.95f;
    sampler->repeat_penalty = 1.0f;
    sampler->repeat_last_n = 64;
    sampler->rng_state = seed ? seed : 0x1234abcd9876ef01ull;
}

static bool seen_recently(const uint32_t *history,
                          size_t history_count,
                          uint32_t id,
                          uint32_t last_n) {
    if (!history || !history_count || !last_n) {
        return false;
    }

    const size_t begin =
        history_count > last_n ? history_count - last_n : 0;

    for (size_t i = begin; i < history_count; ++i) {
        if (history[i] == id) {
            return true;
        }
    }

    return false;
}

static float adjusted_logit(const Sampler *sampler,
                            float logit,
                            uint32_t id,
                            const uint32_t *history,
                            size_t history_count) {
    if (sampler->repeat_penalty > 1.0f &&
        seen_recently(history,
                      history_count,
                      id,
                      sampler->repeat_last_n)) {
        return logit < 0.0f
                   ? logit * sampler->repeat_penalty
                   : logit / sampler->repeat_penalty;
    }

    return logit;
}

static int compare_candidates(const void *left, const void *right) {
    const Candidate *a = (const Candidate *)left;
    const Candidate *b = (const Candidate *)right;

    if (a->logit > b->logit) {
        return -1;
    }
    if (a->logit < b->logit) {
        return 1;
    }
    return 0;
}

uint32_t sampler_sample(Sampler *sampler,
                        const float *logits,
                        uint32_t vocab_size,
                        const uint32_t *history,
                        size_t history_count) {
    if (!vocab_size) {
        return 0;
    }

    if (sampler->temperature <= 0.0f) {
        uint32_t best_id = 0;
        float best_logit = adjusted_logit(sampler,
                                          logits[0],
                                          0,
                                          history,
                                          history_count);

        for (uint32_t id = 1; id < vocab_size; ++id) {
            const float logit = adjusted_logit(sampler,
                                                logits[id],
                                                id,
                                                history,
                                                history_count);
            if (logit > best_logit) {
                best_logit = logit;
                best_id = id;
            }
        }

        return best_id;
    }

    Candidate *candidates = (Candidate *)chris_xmalloc(
        (size_t)vocab_size * sizeof(Candidate));
    const float inverse_temperature = 1.0f / sampler->temperature;

    for (uint32_t id = 0; id < vocab_size; ++id) {
        candidates[id].id = id;
        candidates[id].logit = adjusted_logit(sampler,
                                               logits[id],
                                               id,
                                               history,
                                               history_count) *
                               inverse_temperature;
        candidates[id].probability = 0.0f;
    }

    qsort(candidates,
          vocab_size,
          sizeof(*candidates),
          compare_candidates);

    uint32_t keep = sampler->top_k
                        ? CHRIS_MIN(sampler->top_k, vocab_size)
                        : vocab_size;

    const float max_logit = candidates[0].logit;
    double normalizer = 0.0;

    for (uint32_t i = 0; i < keep; ++i) {
        candidates[i].probability =
            expf(candidates[i].logit - max_logit);
        normalizer += candidates[i].probability;
    }

    for (uint32_t i = 0; i < keep; ++i) {
        candidates[i].probability =
            (float)(candidates[i].probability / normalizer);
    }

    if (sampler->top_p > 0.0f && sampler->top_p < 1.0f) {
        double cumulative = 0.0;
        uint32_t top_p_keep = 0;

        for (; top_p_keep < keep; ++top_p_keep) {
            cumulative += candidates[top_p_keep].probability;
            if (cumulative >= sampler->top_p) {
                ++top_p_keep;
                break;
            }
        }

        if (top_p_keep == 0) {
            top_p_keep = 1;
        }
        keep = top_p_keep;

        normalizer = 0.0;
        for (uint32_t i = 0; i < keep; ++i) {
            normalizer += candidates[i].probability;
        }
        for (uint32_t i = 0; i < keep; ++i) {
            candidates[i].probability =
                (float)(candidates[i].probability / normalizer);
        }
    }

    const double sample = rng_unit(&sampler->rng_state);
    double cumulative = 0.0;
    uint32_t chosen = candidates[keep - 1].id;

    for (uint32_t i = 0; i < keep; ++i) {
        cumulative += candidates[i].probability;
        if (sample <= cumulative) {
            chosen = candidates[i].id;
            break;
        }
    }

    free(candidates);
    return chosen;
}
