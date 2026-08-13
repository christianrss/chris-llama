#ifndef CHRIS_SAMPLER_H
#define CHRIS_SAMPLER_H

#include "common.h"

typedef struct {
    float temperature;
    uint32_t top_k;         /* 0 disables top-k filtering. */
    float top_p;
    float repeat_penalty;
    uint32_t repeat_last_n;
    uint64_t rng_state;
} Sampler;

void sampler_init(Sampler *sampler, uint64_t seed);
uint32_t sampler_sample(Sampler *sampler,
                        const float *logits,
                        uint32_t vocab_size,
                        const uint32_t *history,
                        size_t history_count);

#endif /* CHRIS_SAMPLER_H */
