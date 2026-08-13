#ifndef CHRIS_GPT2_H
#define CHRIS_GPT2_H

#include "backend.h"
#include "gguf.h"

typedef struct {
    float *data;
    uint64_t ne0;  /* Contiguous dimension; input width for matrices. */
    uint64_t ne1;  /* Row/output count; 1 for vectors. */
    bool owns_data;
} Weight;

typedef struct {
    Weight attn_norm_w;
    Weight attn_norm_b;
    Weight attn_qkv_w;
    Weight attn_qkv_b;
    Weight attn_out_w;
    Weight attn_out_b;
    Weight ffn_norm_w;
    Weight ffn_norm_b;
    Weight ffn_up_w;
    Weight ffn_up_b;
    Weight ffn_down_w;
    Weight ffn_down_b;
} GPT2Layer;

typedef struct {
    uint32_t vocab_size;
    uint32_t context_length;
    uint32_t embedding_length;
    uint32_t block_count;
    uint32_t head_count;
    uint32_t feed_forward_length;
    float layer_norm_epsilon;
} GPT2Config;

typedef struct {
    GPT2Config cfg;

    Weight token_embd;
    Weight position_embd;
    Weight output_norm_w;
    Weight output_norm_b;
    Weight output;
    GPT2Layer *layers;

    /* [layer][position][embedding] */
    float *k_cache;
    float *v_cache;

    /* Scratch storage reused by gpt2_forward_token(). */
    float *x;
    float *norm;
    float *qkv;
    float *attn;
    float *proj;
    float *ff;
    float *scores;

    ChrisBackend *backend;
} GPT2Model;

int gpt2_load(GPT2Model *model, const GGUFFile *gguf);
void gpt2_free(GPT2Model *model);
void gpt2_reset(GPT2Model *model);

/* Attach a compute backend and preload matvec weights when supported. */
int gpt2_attach_backend(GPT2Model *model, ChrisBackend *backend);

/*
 * Evaluate one token at position `position`.
 *
 * Pass logits=NULL while ingesting prompt tokens whose logits are not needed.
 */
int gpt2_forward_token(GPT2Model *model,
                       uint32_t token,
                       uint32_t position,
                       float *logits);

void gpt2_print_config(const GPT2Model *model);

#endif /* CHRIS_GPT2_H */
