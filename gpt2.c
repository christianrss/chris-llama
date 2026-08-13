#include "gpt2.h"

static void weight_free(Weight *weight) {
    if (!weight) {
        return;
    }

    if (weight->owns_data) {
        free(weight->data);
    }

    memset(weight, 0, sizeof(*weight));
}

static int weight_load(const GGUFFile *gguf,
                       const char *name,
                       Weight *weight) {
    const GGUFTensorInfo *tensor = gguf_find_tensor(gguf, name);
    if (!tensor) {
        fprintf(stderr, "error: required tensor is missing: %s\n", name);
        return -1;
    }

    if (tensor->n_dims > 2) {
        fprintf(stderr,
                "error: tensor %s has %u dimensions; expected <= 2\n",
                name,
                tensor->n_dims);
        return -1;
    }

    const void *source = gguf_tensor_data(gguf, tensor);
    if (!source) {
        fprintf(stderr, "error: invalid tensor data for %s\n", name);
        return -1;
    }

    if (!quant_nbytes(tensor->type, tensor->n_elements)) {
        fprintf(stderr,
                "error: %s uses unsupported GGML type %u\n",
                name,
                tensor->type);
        return -1;
    }

    weight->data = (float *)chris_xmalloc(
        (size_t)tensor->n_elements * sizeof(float));
    weight->ne0 = tensor->dims[0];
    weight->ne1 = tensor->n_dims > 1 ? tensor->dims[1] : 1;
    weight->owns_data = true;

    if (quant_to_f32(tensor->type,
                     source,
                     tensor->n_elements,
                     weight->data) != 0) {
        fprintf(stderr,
                "error: failed to convert %s (%s) to F32\n",
                name,
                quant_type_name(tensor->type));
        weight_free(weight);
        return -1;
    }

    return 0;
}

/* Return 1 when the optional tensor is not present. */
static int weight_load_optional(const GGUFFile *gguf,
                                const char *name,
                                Weight *weight) {
    if (!gguf_find_tensor(gguf, name)) {
        return 1;
    }
    return weight_load(gguf, name, weight);
}

static int expect_vector(const char *name,
                         const Weight *weight,
                         uint32_t length) {
    if (weight->ne0 == length && weight->ne1 == 1) {
        return 0;
    }

    fprintf(stderr,
            "error: shape of %s = [%" PRIu64 ",%" PRIu64
            "], expected [%u]\n",
            name,
            weight->ne0,
            weight->ne1,
            length);
    return -1;
}

static int expect_matrix(const char *name,
                         const Weight *weight,
                         uint32_t in_features,
                         uint32_t out_features) {
    if (weight->ne0 == in_features && weight->ne1 == out_features) {
        return 0;
    }

    fprintf(stderr,
            "error: shape of %s = [%" PRIu64 ",%" PRIu64
            "], expected [%u,%u]\n",
            name,
            weight->ne0,
            weight->ne1,
            in_features,
            out_features);
    return -1;
}

static void layer_tensor_name(char *buffer,
                              size_t capacity,
                              uint32_t layer,
                              const char *suffix) {
    snprintf(buffer, capacity, "blk.%u.%s", layer, suffix);
}

static int get_config_u32(const GGUFFile *gguf,
                          const char *key,
                          uint32_t *out) {
    uint64_t value = 0;
    if (gguf_get_u64(gguf, key, &value) != 0 || value > UINT32_MAX) {
        fprintf(stderr, "error: missing or invalid metadata: %s\n", key);
        return -1;
    }

    *out = (uint32_t)value;
    return 0;
}

static int load_layer(const GGUFFile *gguf,
                      uint32_t layer_index,
                      GPT2Layer *layer) {
    char name[128];

#define LOAD_LAYER_WEIGHT(SUFFIX, FIELD)                                      \
    do {                                                                       \
        layer_tensor_name(name, sizeof(name), layer_index, SUFFIX);            \
        if (weight_load(gguf, name, &layer->FIELD) != 0) {                     \
            return -1;                                                         \
        }                                                                      \
    } while (0)

    LOAD_LAYER_WEIGHT("attn_norm.weight", attn_norm_w);
    LOAD_LAYER_WEIGHT("attn_norm.bias", attn_norm_b);
    LOAD_LAYER_WEIGHT("attn_qkv.weight", attn_qkv_w);
    LOAD_LAYER_WEIGHT("attn_qkv.bias", attn_qkv_b);
    LOAD_LAYER_WEIGHT("attn_output.weight", attn_out_w);
    LOAD_LAYER_WEIGHT("attn_output.bias", attn_out_b);
    LOAD_LAYER_WEIGHT("ffn_norm.weight", ffn_norm_w);
    LOAD_LAYER_WEIGHT("ffn_norm.bias", ffn_norm_b);
    LOAD_LAYER_WEIGHT("ffn_up.weight", ffn_up_w);
    LOAD_LAYER_WEIGHT("ffn_up.bias", ffn_up_b);
    LOAD_LAYER_WEIGHT("ffn_down.weight", ffn_down_w);
    LOAD_LAYER_WEIGHT("ffn_down.bias", ffn_down_b);

#undef LOAD_LAYER_WEIGHT

    return 0;
}

static int validate_layer(const GPT2Layer *layer,
                          uint32_t embedding_length,
                          uint32_t feed_forward_length) {
    const uint32_t c = embedding_length;
    const uint32_t f = feed_forward_length;

    if (expect_vector("attn_norm.weight", &layer->attn_norm_w, c) ||
        expect_vector("attn_norm.bias", &layer->attn_norm_b, c) ||
        expect_matrix("attn_qkv.weight", &layer->attn_qkv_w, c, 3 * c) ||
        expect_vector("attn_qkv.bias", &layer->attn_qkv_b, 3 * c) ||
        expect_matrix("attn_output.weight", &layer->attn_out_w, c, c) ||
        expect_vector("attn_output.bias", &layer->attn_out_b, c) ||
        expect_vector("ffn_norm.weight", &layer->ffn_norm_w, c) ||
        expect_vector("ffn_norm.bias", &layer->ffn_norm_b, c) ||
        expect_matrix("ffn_up.weight", &layer->ffn_up_w, c, f) ||
        expect_vector("ffn_up.bias", &layer->ffn_up_b, f) ||
        expect_matrix("ffn_down.weight", &layer->ffn_down_w, f, c) ||
        expect_vector("ffn_down.bias", &layer->ffn_down_b, c)) {
        return -1;
    }

    return 0;
}

int gpt2_load(GPT2Model *model, const GGUFFile *gguf) {
    memset(model, 0, sizeof(*model));

    const char *architecture = gguf_get_string(gguf, "general.architecture");
    if (!architecture || strcmp(architecture, "gpt2") != 0) {
        fprintf(stderr,
                "error: this runtime requires general.architecture='gpt2' "
                "(file: %s)\n",
                architecture ? architecture : "<missing>");
        return -1;
    }

    GPT2Config *config = &model->cfg;
    if (get_config_u32(gguf,
                       "gpt2.context_length",
                       &config->context_length) ||
        get_config_u32(gguf,
                       "gpt2.embedding_length",
                       &config->embedding_length) ||
        get_config_u32(gguf,
                       "gpt2.block_count",
                       &config->block_count) ||
        get_config_u32(gguf,
                       "gpt2.attention.head_count",
                       &config->head_count)) {
        return -1;
    }

    /* Older converters may omit this field. GPT-2 uses a 4x MLP expansion. */
    uint64_t feed_forward = 0;
    if (gguf_get_u64(gguf,
                     "gpt2.feed_forward_length",
                     &feed_forward) == 0 &&
        feed_forward <= UINT32_MAX) {
        config->feed_forward_length = (uint32_t)feed_forward;
    } else {
        config->feed_forward_length = 4 * config->embedding_length;
    }

    double layer_norm_epsilon = 1e-5;
    if (gguf_get_f64(gguf,
                     "gpt2.attention.layer_norm_epsilon",
                     &layer_norm_epsilon) != 0) {
        layer_norm_epsilon = 1e-5;
    }
    config->layer_norm_epsilon = (float)layer_norm_epsilon;

    if (!config->context_length ||
        !config->embedding_length ||
        !config->block_count ||
        !config->head_count ||
        !config->feed_forward_length) {
        fprintf(stderr,
                "error: GPT-2 configuration contains a zero-sized dimension\n");
        return -1;
    }

    if (config->embedding_length % config->head_count != 0) {
        fprintf(stderr,
                "error: embedding_length must be divisible by head_count\n");
        return -1;
    }

    fprintf(stderr, "[load] embeddings...\n");
    if (weight_load(gguf, "token_embd.weight", &model->token_embd) ||
        weight_load(gguf, "position_embd.weight", &model->position_embd)) {
        goto fail;
    }

    config->vocab_size = (uint32_t)model->token_embd.ne1;
    if (expect_matrix("token_embd.weight",
                      &model->token_embd,
                      config->embedding_length,
                      config->vocab_size) ||
        expect_matrix("position_embd.weight",
                      &model->position_embd,
                      config->embedding_length,
                      config->context_length)) {
        goto fail;
    }

    fprintf(stderr,
            "[load] %u transformer blocks...\n",
            config->block_count);

    model->layers = (GPT2Layer *)chris_xcalloc(
        config->block_count,
        sizeof(GPT2Layer));

    for (uint32_t i = 0; i < config->block_count; ++i) {
        if (load_layer(gguf, i, &model->layers[i]) ||
            validate_layer(&model->layers[i],
                           config->embedding_length,
                           config->feed_forward_length)) {
            goto fail;
        }

        if ((i + 1) % 4 == 0 || i + 1 == config->block_count) {
            fprintf(stderr, "  loaded %u/%u\n", i + 1, config->block_count);
        }
    }

    if (weight_load(gguf, "output_norm.weight", &model->output_norm_w) ||
        weight_load(gguf, "output_norm.bias", &model->output_norm_b)) {
        goto fail;
    }

    if (expect_vector("output_norm.weight",
                      &model->output_norm_w,
                      config->embedding_length) ||
        expect_vector("output_norm.bias",
                      &model->output_norm_b,
                      config->embedding_length)) {
        goto fail;
    }

    const int output_result =
        weight_load_optional(gguf, "output.weight", &model->output);
    if (output_result < 0) {
        goto fail;
    }

    if (output_result == 1) {
        /* GPT-2 normally ties the LM head to the token embedding matrix. */
        model->output = model->token_embd;
        model->output.owns_data = false;
    }

    if (expect_matrix("output.weight",
                      &model->output,
                      config->embedding_length,
                      config->vocab_size)) {
        goto fail;
    }

    const size_t c = config->embedding_length;
    const size_t f = config->feed_forward_length;
    const size_t layers = config->block_count;
    const size_t context = config->context_length;

    if (layers && context > SIZE_MAX / layers / c) {
        fprintf(stderr, "error: KV cache size exceeds SIZE_MAX\n");
        goto fail;
    }

    const size_t kv_elements = layers * context * c;
    model->k_cache = (float *)chris_xcalloc(kv_elements, sizeof(float));
    model->v_cache = (float *)chris_xcalloc(kv_elements, sizeof(float));
    model->x = (float *)chris_xmalloc(c * sizeof(float));
    model->norm = (float *)chris_xmalloc(c * sizeof(float));
    model->qkv = (float *)chris_xmalloc(3 * c * sizeof(float));
    model->attn = (float *)chris_xmalloc(c * sizeof(float));
    model->proj = (float *)chris_xmalloc(c * sizeof(float));
    model->ff = (float *)chris_xmalloc(f * sizeof(float));
    model->scores = (float *)chris_xmalloc(context * sizeof(float));

    return 0;

fail:
    gpt2_free(model);
    return -1;
}

void gpt2_free(GPT2Model *model) {
    if (!model) {
        return;
    }

    weight_free(&model->token_embd);
    weight_free(&model->position_embd);
    weight_free(&model->output_norm_w);
    weight_free(&model->output_norm_b);
    weight_free(&model->output);

    if (model->layers) {
        for (uint32_t i = 0; i < model->cfg.block_count; ++i) {
            GPT2Layer *layer = &model->layers[i];
            weight_free(&layer->attn_norm_w);
            weight_free(&layer->attn_norm_b);
            weight_free(&layer->attn_qkv_w);
            weight_free(&layer->attn_qkv_b);
            weight_free(&layer->attn_out_w);
            weight_free(&layer->attn_out_b);
            weight_free(&layer->ffn_norm_w);
            weight_free(&layer->ffn_norm_b);
            weight_free(&layer->ffn_up_w);
            weight_free(&layer->ffn_up_b);
            weight_free(&layer->ffn_down_w);
            weight_free(&layer->ffn_down_b);
        }
    }

    free(model->layers);
    free(model->k_cache);
    free(model->v_cache);
    free(model->x);
    free(model->norm);
    free(model->qkv);
    free(model->attn);
    free(model->proj);
    free(model->ff);
    free(model->scores);

    memset(model, 0, sizeof(*model));
}

void gpt2_reset(GPT2Model *model) {
    if (!model || !model->k_cache) {
        return;
    }

    const size_t cache_elements =
        (size_t)model->cfg.block_count *
        model->cfg.context_length *
        model->cfg.embedding_length;

    memset(model->k_cache, 0, cache_elements * sizeof(float));
    memset(model->v_cache, 0, cache_elements * sizeof(float));
}

static void layer_norm(float *out,
                       const float *x,
                       const Weight *weight,
                       const Weight *bias,
                       uint32_t length,
                       float epsilon) {
    double sum = 0.0;
    double sum_squared = 0.0;

    for (uint32_t i = 0; i < length; ++i) {
        sum += x[i];
        sum_squared += (double)x[i] * x[i];
    }

    const double mean = sum / length;
    double variance = sum_squared / length - mean * mean;
    if (variance < 0.0) {
        variance = 0.0;
    }

    const float inverse_std =
        1.0f / sqrtf((float)variance + epsilon);

    for (uint32_t i = 0; i < length; ++i) {
        out[i] = (x[i] - (float)mean) *
                 inverse_std *
                 weight->data[i] +
                 bias->data[i];
    }
}

/* All linear projections go through one backend boundary. */
static int matvec(GPT2Model *model,
                  float *out,
                  const Weight *weight,
                  const float *x,
                  const Weight *bias) {
    if (!model->backend) {
        fprintf(stderr,
                "error: no compute backend is attached to the model\n");
        return -1;
    }

    return chris_backend_matvec(model->backend,
                                out,
                                weight->data,
                                x,
                                bias ? bias->data : NULL,
                                (uint32_t)weight->ne0,
                                (uint32_t)weight->ne1);
}

static inline float gelu_new(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float cubic = x * x * x;
    return 0.5f * x *
           (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * cubic)));
}

static void attention_incremental(GPT2Model *model,
                                  uint32_t layer_index,
                                  uint32_t position,
                                  const float *q,
                                  const float *k,
                                  const float *v) {
    const uint32_t embedding = model->cfg.embedding_length;
    const uint32_t heads = model->cfg.head_count;
    const uint32_t head_dim = embedding / heads;
    const uint32_t context = model->cfg.context_length;

    const size_t current_offset =
        ((size_t)layer_index * context + position) * embedding;

    memcpy(model->k_cache + current_offset,
           k,
           embedding * sizeof(float));
    memcpy(model->v_cache + current_offset,
           v,
           embedding * sizeof(float));

    /*
     * Attention stays on the host for now. Keeping this code straightforward
     * makes CPU/SYCL result comparisons easier while the large projections are
     * moved to the backend first.
     */
    for (uint32_t head = 0; head < heads; ++head) {
        const float *query = q + (size_t)head * head_dim;
        float max_score = -INFINITY;

        for (uint32_t t = 0; t <= position; ++t) {
            const size_t cache_offset =
                ((size_t)layer_index * context + t) * embedding +
                (size_t)head * head_dim;
            const float *key = model->k_cache + cache_offset;

            float score = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                score += query[d] * key[d];
            }

            score /= sqrtf((float)head_dim);
            model->scores[t] = score;
            if (score > max_score) {
                max_score = score;
            }
        }

        float normalizer = 0.0f;
        for (uint32_t t = 0; t <= position; ++t) {
            model->scores[t] = expf(model->scores[t] - max_score);
            normalizer += model->scores[t];
        }

        const float inverse_normalizer = 1.0f / normalizer;
        float *output = model->attn + (size_t)head * head_dim;
        memset(output, 0, head_dim * sizeof(float));

        for (uint32_t t = 0; t <= position; ++t) {
            const float attention_weight =
                model->scores[t] * inverse_normalizer;
            const size_t cache_offset =
                ((size_t)layer_index * context + t) * embedding +
                (size_t)head * head_dim;
            const float *value = model->v_cache + cache_offset;

            for (uint32_t d = 0; d < head_dim; ++d) {
                output[d] += attention_weight * value[d];
            }
        }
    }
}

static int preload_weight(ChrisBackend *backend, const Weight *weight) {
    if (!weight || !weight->data || !weight->ne0 || !weight->ne1) {
        return 0;
    }

    if (weight->ne1 > SIZE_MAX / weight->ne0) {
        return -1;
    }

    return chris_backend_preload(
        backend,
        weight->data,
        (size_t)(weight->ne0 * weight->ne1));
}

int gpt2_attach_backend(GPT2Model *model, ChrisBackend *backend) {
    if (!model || !backend) {
        return -1;
    }

    model->backend = backend;

    fprintf(stderr,
            "[backend] preloading projection weights to %s...\n",
            chris_backend_device_name(backend));

    for (uint32_t i = 0; i < model->cfg.block_count; ++i) {
        GPT2Layer *layer = &model->layers[i];

        if (preload_weight(backend, &layer->attn_qkv_w) ||
            preload_weight(backend, &layer->attn_qkv_b) ||
            preload_weight(backend, &layer->attn_out_w) ||
            preload_weight(backend, &layer->attn_out_b) ||
            preload_weight(backend, &layer->ffn_up_w) ||
            preload_weight(backend, &layer->ffn_up_b) ||
            preload_weight(backend, &layer->ffn_down_w) ||
            preload_weight(backend, &layer->ffn_down_b)) {
            return -1;
        }
    }

    if (preload_weight(backend, &model->output)) {
        return -1;
    }

    fprintf(stderr, "[backend] preload complete\n");
    return 0;
}

int gpt2_forward_token(GPT2Model *model,
                       uint32_t token,
                       uint32_t position,
                       float *logits) {
    const uint32_t embedding = model->cfg.embedding_length;
    const uint32_t feed_forward = model->cfg.feed_forward_length;

    if (token >= model->cfg.vocab_size ||
        position >= model->cfg.context_length) {
        return -1;
    }

    const float *token_embedding =
        model->token_embd.data + (size_t)token * embedding;
    const float *position_embedding =
        model->position_embd.data + (size_t)position * embedding;

    for (uint32_t i = 0; i < embedding; ++i) {
        model->x[i] = token_embedding[i] + position_embedding[i];
    }

    for (uint32_t layer_index = 0;
         layer_index < model->cfg.block_count;
         ++layer_index) {
        GPT2Layer *layer = &model->layers[layer_index];

        /* Pre-LN attention block: x = x + Attn(LN(x)). */
        layer_norm(model->norm,
                   model->x,
                   &layer->attn_norm_w,
                   &layer->attn_norm_b,
                   embedding,
                   model->cfg.layer_norm_epsilon);

        if (matvec(model,
                   model->qkv,
                   &layer->attn_qkv_w,
                   model->norm,
                   &layer->attn_qkv_b)) {
            return -1;
        }

        const float *q = model->qkv;
        const float *k = model->qkv + embedding;
        const float *v = model->qkv + 2 * embedding;

        attention_incremental(model,
                              layer_index,
                              position,
                              q,
                              k,
                              v);

        if (matvec(model,
                   model->proj,
                   &layer->attn_out_w,
                   model->attn,
                   &layer->attn_out_b)) {
            return -1;
        }

        for (uint32_t i = 0; i < embedding; ++i) {
            model->x[i] += model->proj[i];
        }

        /* Pre-LN MLP block: Linear -> GELU(new) -> Linear -> residual. */
        layer_norm(model->norm,
                   model->x,
                   &layer->ffn_norm_w,
                   &layer->ffn_norm_b,
                   embedding,
                   model->cfg.layer_norm_epsilon);

        if (matvec(model,
                   model->ff,
                   &layer->ffn_up_w,
                   model->norm,
                   &layer->ffn_up_b)) {
            return -1;
        }

        for (uint32_t i = 0; i < feed_forward; ++i) {
            model->ff[i] = gelu_new(model->ff[i]);
        }

        if (matvec(model,
                   model->proj,
                   &layer->ffn_down_w,
                   model->ff,
                   &layer->ffn_down_b)) {
            return -1;
        }

        for (uint32_t i = 0; i < embedding; ++i) {
            model->x[i] += model->proj[i];
        }
    }

    if (logits) {
        layer_norm(model->norm,
                   model->x,
                   &model->output_norm_w,
                   &model->output_norm_b,
                   embedding,
                   model->cfg.layer_norm_epsilon);

        if (matvec(model,
                   logits,
                   &model->output,
                   model->norm,
                   NULL)) {
            return -1;
        }
    }

    return 0;
}

void gpt2_print_config(const GPT2Model *model) {
    fprintf(stderr, "GPT-2 config:\n");
    fprintf(stderr, "  vocab_size          %u\n", model->cfg.vocab_size);
    fprintf(stderr,
            "  context_length      %u\n",
            model->cfg.context_length);
    fprintf(stderr,
            "  embedding_length    %u\n",
            model->cfg.embedding_length);
    fprintf(stderr, "  block_count         %u\n", model->cfg.block_count);
    fprintf(stderr, "  head_count          %u\n", model->cfg.head_count);
    fprintf(stderr,
            "  feed_forward_length %u\n",
            model->cfg.feed_forward_length);
    fprintf(stderr,
            "  layer_norm_epsilon  %.9g\n",
            model->cfg.layer_norm_epsilon);
}
