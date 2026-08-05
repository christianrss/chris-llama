#include "common.h"
#include "tensor.h"
#include "kv_cache.h"
#include "quant.h"
#include "gguf.h"
#include "model.h"
#include "tokenizer.h"
#include "generate.h"

/**
 * Unit test for matrix multiplication core tensor math
 */
void test_matmul()
{
    printf("\n[MatMul Unit Test]\n");
    // Define matrix shapes: A(2x2), B(2x2), output C(2x2)
    u32 shape_a[] = {2, 2};
    u32 shape_b[] = {2, 2};
    u32 shape_c[] = {2, 2};

    // Create empty tensors
    Tensor* A = tensor_create(2, shape_a);
    Tensor* B = tensor_create(2, shape_b);
    Tensor* C = tensor_create(2, shape_c);

    // Fill test matrix values
    // A = [[1, 2], [3, 4]]
    A->data[0] = 1.0f; A->data[1] = 2.0f;
    A->data[2] = 3.0f; A->data[3] = 4.0f;
    // B = [[5, 6], [7, 8]]
    B->data[0] = 5.0f; B->data[1] = 6.0f;
    B->data[2] = 7.0f; B->data[3] = 8.0f;

    // Execute matrix multiplication
    matmul(A, B, C);

    // Print computed result matrix
    printf("%.2f %.2f\n", C->data[0], C->data[1]);
    printf("%.2f %.2f\n", C->data[2], C->data[3]);

    // Free allocated tensor memory
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
}

/**
 * Unit test to verify KV Cache init and reset logic
 */
void test_kv_cache()
{
    printf("\n[KV Cache Unit Test]\n");
    KVCache cache;

    // Initialize cache with hidden dim 512, max seqence from global macro
    kv_cache_init(&cache, 512, MAX_SEQ_LEN);

    // Simulate storing tokens
    cache.cur_seq = 10;
    printf("Cached token count before reset: %u\n", cache.cur_seq);

    // Execute full cache reset
    kv_cache_reset(&cache);
    printf("Cached token count after reset: %u\n", cache.cur_seq);
}

/**
 * Unit testfor INT4 quantization & dequantization correctness
 * Compare original values with restored decompressed values
 */
void test_int4_quant()
{
    printf("\n[INT4 Quantization Unit Test]\n");
    // Sample input float vector
    f32 original[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const u32 elem_count = 8;
    u8 compressed_buf[elem_count / 2]; // 2 int4 per byte
    f32 restored[elem_count];
    f32 scale_param, zp_param;

    // Step 1: Quantize FP32 to packed INT4 bytes
    quant_int4(original, compressed_buf, elem_count, &scale_param, &zp_param);
    // Step 2: Decompress INT4 back to FP32
    dequant_int4(restored, compressed_buf, elem_count, scale_param, zp_param);

    // Print comparison of original vs recovered values
    printf("Original | Restored\n");
    for (u32 i = 0; i < elem_count; i++)
    {
        printf("%.2f      %.2f\n", original[i], restored[i]);
    }
}

/**
 * Minimal GGUF loader test: only validate header parsing
 * Pass your tinyllama .gguf filepath as argument to run
 */
void test_gguf_loader(const char* model_path)
{
    printf("\n[GGUF mmap Loader Unit Test]\n");
    GGUFFile gf;
    int ret = gguf_open(model_path, &gf);
    if (ret != 0)
    {
        printf("GGUF load test FAILED, invalid file path or format\n");
        return;
    }

    printf("GGUF file loaded successfully\n");
    printf("GGUF Version: %u\n", gf.hdr.version);
    printf("Total tensors in model: %llu\n", (unsigned long long)gf.hdr.n_tensors);
    printf("Total metadata entries: %llu\n", (unsigned long long)gf.hdr.n_metadata);

    gguf_close(&gf);
    printf("GGUF resource claned up\n");
}

/**
 * Unit test for RMSNorm normalization calculation
 */
void test_rms_norm()
{
    printf("\n[RMSNorm Unit Test]\n");
    // Sample input vector
    f32 x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    // Uniform weight vector for simple testing
    f32 w[] = {1.0f, 1.0f, 1.0f, 1.0f};
    f32 out[4];

    // Run RMSNorm calculation
    rms_norm(out, x, w, 4);

    // Print normalized output values
    printf("Normalized vector: ");
    for (int i = 0; i < 4; i++)
        printf("%.4f ", out[i]);
    printf("\n");
}

/**
 * Unit test for SwiGLU feed-forward activation
 */
void test_swiglu()
{
    printf("\n[SwiGLU Unit Test]\n");
    // Sample input vectors for gate and up
    f32 gate[] = {1.0f, -1.0f, 2.0f, -2.0f};
    f32 up[]   = {2.0f,  3.0f, 1.0f,  4.0f};
    f32 out[4];

    // Run SwiGLU calculation
    swiglu(out, gate, up, 4);
    printf("SwiGLU ouput vector: ");
    for (int i = 0; i < 4; i++)
        printf("%.4f ", out[i]);
    printf("\n");
}

/**
 * Unit test for RoPE rotary positional encoding transformation
 */
void test_rope()
{
    printf("\n[RoPE Rotary Positional Encoding Test]\n");
    // Short test vector, total dim = 8, single head dim = 8
    f32 q[] = {1, 0, 1, 0, 1, 0, 1, 0};
    f32 k[] = {1, 0, 1, 0, 1, 0, 1, 0};
    u32 pos = 5;
    u32 dim = 8;
    u32 head_dim = 8;

    // Apply RoPE transformation to Q and K vectors
    rope(q, k, pos, dim, head_dim);

    printf("Rotated Q vector: ");
    for(int i = 0; i < 8; i++) {
        printf("%.3f ", q[i]);
    }
    printf("\n");
}

/**
 * Unit test for Causal Multi-Head Attention with KV Cache
 */
void test_causal_mha()
{
    // Print unit test header to console
    printf("\n[Causal Multi-Head Attention Unit Test]\n");
    // Define tensor shape: single token, hidden dimension = 8
    u32 vec_shape[] = {1, 8};
    // Allocate tensor for query vector
    Tensor* q = tensor_create(2, vec_shape);
    // Allocate tensor for key vector
    Tensor* k = tensor_create(2, vec_shape);
    // Allocate tensor for value vector
    Tensor* v = tensor_create(2, vec_shape);
    // Allocate tensor to store attention output result
    Tensor* out = tensor_create(2, vec_shape);

    // Fill sequential numeric test data to Q/K/V tensors
    for (u32 i = 0; i < 8; i++)
    {
        q->data[i] = (f32)i;
        k->data[i] = (f32)i;
        v->data[i] = (f32)i;
    }

    // Declare KV Cache storage structure
    KVCache cache;
    // Initialize KV Cache with hidden dim 8 and max sequence length
    kv_cache_init(&cache, 8, MAX_SEQ_LEN);

    // Run causal MHA forward pass for token at position 0, single attention head
    causal_mha(q, k, v, &cache, out, 0, 1);
    // Print computed attention output vector of position 0
    printf("Attention output at pos 0: ");
    for (int i = 0; i < 8; i++)
    {
        printf("%.2f ", out->data[i]);
    }
    printf("\n");

    // Free dinamically allocated tensor memory to avoid memory leaks
    tensor_free(q);
    tensor_free(k);
    tensor_free(v);
    tensor_free(out);
    // Clear all cached key and value states for next test
    kv_cache_reset(&cache);
}

/**
 * Unit test for test <-> token conversion 
 * Test the full tokenizer encode and decode workflow
 */
void test_tokenizer()
{
    // Print test section title on console
    printf("\n[Tokenizer Unit Test]\n");
    // Sample input sentence for tokenization test
    const char* prompt = "hello world how are you";
    // Buffer to store converted integer token IDs
    u32 tokens[64];
    // Convert plain text string into token ID array
    u32 token_cnt = text_to_tokens(prompt, tokens, 64);
    // Print original input text and total number of generated tokens
    printf("Input text: %s\nToken count: %u\nToken IDs: ", prompt, token_cnt);
    // Loop to print every token ID in sequence
    for (u32 i = 0; i < token_cnt; i++)
        printf("%u ", tokens[i]);
    printf("\n");

    // Demo of token decoding: convert single token ID back to readable word
    char buf[32];
    // Decode the scond token (index 1, BOS is index 0)
    token_to_text(tokens[1], buf, 32);
    // Print decoded text matched with corresponding token ID
    printf("Token %u decode text: %s\n", tokens[1], buf);
}

/**
 * End-to-end autoregressive generation test
 * Full pipeline test: tokenizer -> model forward -> autoregressive token generation
 */
void test_autoregressive_generate()
{
    // Print test section header for generation pipleine
    printf("\n[Autoregressive Generation End-To-End Test]\n");
    // Static hyperparameter configuration matching TinyLlama official setting
    LLaMAConfig cfg = {
        .dim = 512,             // Model hidden embedding dimension
        .n_layers = 22,         // Total of transformer decoder layers 
        .n_heads = 32,          // Number of multi-head attention heads
        .vocab_size = 32000,    // Total vocabulary size of LLaMA tokenizer
        .seq_len = MAX_SEQ_LEN  // Maximum supported context sequence length
    };

    // Declaremain LLM model instance
    LLaMAModel model;
    // Initialize all model tensors and layers with above config
    llama_model_init(&model, &cfg);

    // Dclare KV cache instance for fast generation
    KVCache cache;
    // Allocate memory for key/value cache storage
    kv_cache_init(&cache, cfg.dim, cfg.seq_len);

    // Raw user input prompt for chat test
    const char* user_prompt = "hello llama";
    // Buffer to hold encoded input prompt token IDs
    u32 input_tokens[256];
    // Convert input text into token sequence
    u32 in_cnt = text_to_tokens(user_prompt, input_tokens, 256);

    // Output buffer storing full prompt + newly generated tokens
    u32 output_tokens[512];
    // Run complete autoregressive generation, limit max 10 newly generated tokens
    u32 total = generate_autoregressive(&model, &cache, input_tokens, in_cnt, output_tokens, 10);

    // Print total length of combined prompt + generate token sequence
    printf("Generated total token length: %u\nToken sequence: ", total);
    // Print every token ID in the full output sequence
    for(u32 i = 0; i < total; i++)
        printf("%u ", output_tokens[i]);
    printf("\n");
}


int main(int argc, char** argv)
{
    printf("===== Section 9: Tokenizer & Autoregressive Generation Full Test Suite  =====\n");
    test_matmul();
    test_kv_cache();
    test_int4_quant();
    test_rms_norm();
    test_swiglu();
    test_rope();
    test_causal_mha();
    test_tokenizer();
    test_autoregressive_generate();

    // If user pass gguf file path, run GGUF test
    if (argc >= 2)
    {
        test_gguf_loader(argv[1]);
    }
    else
    {
        printf("\nHint: Run with ./chris_llama model.gguf to test GGUF loading\n");
    }

    printf("\nAll  tests finished without error.\n");
    return 0;
}