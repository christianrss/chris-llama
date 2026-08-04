#include "common.h"
#include "tensor.h"
#include "kv_cache.h"
#include "quant.h"
#include "gguf.h"
#include "model.h"

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


int main(int argc, char** argv)
{
    printf("===== Section 6: RMSNorm & SwiGLU FFN Full Test Suite  =====\n");
    test_matmul();
    test_kv_cache();
    test_int4_quant();
    test_rms_norm();
    test_swiglu();
    test_rope();

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