#include "quant.h"

/*
 * Reference dequantization path. We expand supported GGML blocks to FP32 once
 * at model load time. It uses more memory than fused quantized kernels, but it
 * keeps the transformer code independent of storage format.
 */

float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t frac = (uint32_t)h & 0x03ffu;
    uint32_t out;

    if (exp == 0) {
        if (frac == 0) {
            out = sign;
        } else {
            /* Normalize an IEEE-754 half subnormal. */
            int shift = 0;
            while ((frac & 0x0400u) == 0) {
                frac <<= 1;
                shift++;
            }
            frac &= 0x03ffu;
            const uint32_t exp32 = (uint32_t)(127 - 15 - shift + 1);
            out = sign | (exp32 << 23) | (frac << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (frac << 13);
    } else {
        const uint32_t exp32 = exp + (127 - 15);
        out = sign | (exp32 << 23) | (frac << 13);
    }

    float f;
    memcpy(&f, &out, sizeof(f));
    return f;
}

typedef struct { uint16_t d; uint8_t qs[16]; } block_q4_0;       /* 32 vals */
typedef struct { uint16_t d, m; uint8_t qs[16]; } block_q4_1;    /* 32 vals */
typedef struct { uint16_t d; uint8_t qh[4], qs[16]; } block_q5_0;
typedef struct { uint16_t d, m; uint8_t qh[4], qs[16]; } block_q5_1;
typedef struct { uint16_t d; int8_t qs[32]; } block_q8_0;

#define QK_K 256
#define K_SCALE_SIZE 12

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K/2];
} block_q4_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K/8];
    uint8_t qs[QK_K/2];
} block_q5_K;

typedef struct {
    uint8_t ql[QK_K/2];
    uint8_t qh[QK_K/4];
    int8_t scales[QK_K/16];
    uint16_t d;
} block_q6_K;

_Static_assert(sizeof(block_q4_0) == 18, "q4_0 layout");
_Static_assert(sizeof(block_q4_1) == 20, "q4_1 layout");
_Static_assert(sizeof(block_q5_0) == 22, "q5_0 layout");
_Static_assert(sizeof(block_q5_1) == 24, "q5_1 layout");
_Static_assert(sizeof(block_q8_0) == 34, "q8_0 layout");
_Static_assert(sizeof(block_q4_K) == 144, "q4_K layout");
_Static_assert(sizeof(block_q5_K) == 176, "q5_K layout");
_Static_assert(sizeof(block_q6_K) == 210, "q6_K layout");

const char *quant_type_name(uint32_t type) {
    switch (type) {
        case CHRIS_GGML_TYPE_F32: return "F32";
        case CHRIS_GGML_TYPE_F16: return "F16";
        case CHRIS_GGML_TYPE_Q4_0: return "Q4_0";
        case CHRIS_GGML_TYPE_Q4_1: return "Q4_1";
        case CHRIS_GGML_TYPE_Q5_0: return "Q5_0";
        case CHRIS_GGML_TYPE_Q5_1: return "Q5_1";
        case CHRIS_GGML_TYPE_Q8_0: return "Q8_0";
        case CHRIS_GGML_TYPE_Q4_K: return "Q4_K";
        case CHRIS_GGML_TYPE_Q5_K: return "Q5_K";
        case CHRIS_GGML_TYPE_Q6_K: return "Q6_K";
        default: return "UNSUPPORTED";
    }
}

size_t quant_nbytes(uint32_t type, uint64_t n) {
    uint64_t qk = 1, bs = 0;
    switch (type) {
        case CHRIS_GGML_TYPE_F32: qk = 1; bs = 4; break;
        case CHRIS_GGML_TYPE_F16: qk = 1; bs = 2; break;
        case CHRIS_GGML_TYPE_Q4_0: qk = 32; bs = sizeof(block_q4_0); break;
        case CHRIS_GGML_TYPE_Q4_1: qk = 32; bs = sizeof(block_q4_1); break;
        case CHRIS_GGML_TYPE_Q5_0: qk = 32; bs = sizeof(block_q5_0); break;
        case CHRIS_GGML_TYPE_Q5_1: qk = 32; bs = sizeof(block_q5_1); break;
        case CHRIS_GGML_TYPE_Q8_0: qk = 32; bs = sizeof(block_q8_0); break;
        case CHRIS_GGML_TYPE_Q4_K: qk = QK_K; bs = sizeof(block_q4_K); break;
        case CHRIS_GGML_TYPE_Q5_K: qk = QK_K; bs = sizeof(block_q5_K); break;
        case CHRIS_GGML_TYPE_Q6_K: qk = QK_K; bs = sizeof(block_q6_K); break;
        default: return 0;
    }
    if (n % qk != 0) return 0;
    return (size_t)(n / qk * bs);
}

static void deq_q4_0(const block_q4_0 *x, uint64_t nb, float *y) {
    for (uint64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < 16; ++j) {
            y[i*32 + j]      = ((int)(x[i].qs[j] & 0x0f) - 8) * d;
            y[i*32 + j + 16] = ((int)(x[i].qs[j] >> 4)   - 8) * d;
        }
    }
}

static void deq_q4_1(const block_q4_1 *x, uint64_t nb, float *y) {
    for (uint64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        const float m = fp16_to_fp32(x[i].m);
        for (int j = 0; j < 16; ++j) {
            y[i*32 + j]      = (x[i].qs[j] & 0x0f) * d + m;
            y[i*32 + j + 16] = (x[i].qs[j] >> 4)   * d + m;
        }
    }
}

static void deq_q5_0(const block_q5_0 *x, uint64_t nb, float *y) {
    for (uint64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        uint32_t qh = 0;
        memcpy(&qh, x[i].qh, sizeof(qh));
        for (int j = 0; j < 16; ++j) {
            const uint8_t h0 = (uint8_t)(((qh >> j) << 4) & 0x10);
            const uint8_t h1 = (uint8_t)((qh >> (j + 12)) & 0x10);
            const int q0 = ((x[i].qs[j] & 0x0f) | h0) - 16;
            const int q1 = ((x[i].qs[j] >> 4) | h1) - 16;
            y[i*32 + j] = q0 * d;
            y[i*32 + j + 16] = q1 * d;
        }
    }
}

static void deq_q5_1(const block_q5_1 *x, uint64_t nb, float *y) {
    for (uint64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        const float m = fp16_to_fp32(x[i].m);
        uint32_t qh = 0;
        memcpy(&qh, x[i].qh, sizeof(qh));
        for (int j = 0; j < 16; ++j) {
            const uint8_t h0 = (uint8_t)(((qh >> j) << 4) & 0x10);
            const uint8_t h1 = (uint8_t)((qh >> (j + 12)) & 0x10);
            const int q0 = (x[i].qs[j] & 0x0f) | h0;
            const int q1 = (x[i].qs[j] >> 4) | h1;
            y[i*32 + j] = q0 * d + m;
            y[i*32 + j + 16] = q1 * d + m;
        }
    }
}

static void deq_q8_0(const block_q8_0 *x, uint64_t nb, float *y) {
    for (uint64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < 32; ++j) y[i*32 + j] = x[i].qs[j] * d;
    }
}

/* Decode the packed 6-bit scale/min pair used by Q4_K and Q5_K. */
static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uint8_t)((q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

static void deq_q4_K(const block_q4_K *x, uint64_t nb, float *y) {
    for (uint64_t i = 0; i < nb; ++i) {
        const uint8_t *q = x[i].qs;
        const float d = fp16_to_fp32(x[i].d);
        const float dmin = fp16_to_fp32(x[i].dmin);
        int is = 0;
        float *out = y + i * QK_K;

        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is, x[i].scales, &sc, &m);
            const float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; ++l) out[j + l]      = d1 * (q[l] & 0x0f) - m1;
            for (int l = 0; l < 32; ++l) out[j + 32 + l] = d2 * (q[l] >> 4)   - m2;
            q += 32;
            is += 2;
        }
    }
}

static void deq_q5_K(const block_q5_K *x, uint64_t nb, float *y) {
    for (uint64_t i = 0; i < nb; ++i) {
        const uint8_t *ql = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const float d = fp16_to_fp32(x[i].d);
        const float dmin = fp16_to_fp32(x[i].dmin);
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        float *out = y + i * QK_K;

        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is, x[i].scales, &sc, &m);
            const float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; ++l) {
                const int q1 = (ql[l] & 0x0f) + ((qh[l] & u1) ? 16 : 0);
                const int q2 = (ql[l] >> 4)   + ((qh[l] & u2) ? 16 : 0);
                out[j + l]      = d1 * q1 - m1;
                out[j + 32 + l] = d2 * q2 - m2;
            }
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

static void deq_q6_K(const block_q6_K *x, uint64_t nb, float *y) {
    for (uint64_t i = 0; i < nb; ++i) {
        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t *sc = x[i].scales;
        const float d = fp16_to_fp32(x[i].d);
        float *out = y + i * QK_K;

        /* A 256-value super-block is stored as two 128-value halves. */
        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = (ql[l]      & 0x0f) | (((qh[l] >> 0) & 3) << 4);
                const int q2 = (ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4);
                const int q3 = (ql[l]      >> 4)   | (((qh[l] >> 4) & 3) << 4);
                const int q4 = (ql[l + 32] >> 4)   | (((qh[l] >> 6) & 3) << 4);
                out[n + l]      = d * sc[is + 0] * (q1 - 32);
                out[n + l + 32] = d * sc[is + 2] * (q2 - 32);
                out[n + l + 64] = d * sc[is + 4] * (q3 - 32);
                out[n + l + 96] = d * sc[is + 6] * (q4 - 32);
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

int quant_to_f32(uint32_t type, const void *src, uint64_t n, float *dst) {
    if (!src || !dst) return -1;
    switch (type) {
        case CHRIS_GGML_TYPE_F32:
            memcpy(dst, src, (size_t)n * sizeof(float));
            return 0;
        case CHRIS_GGML_TYPE_F16: {
            const uint16_t *p = (const uint16_t *)src;
            for (uint64_t i = 0; i < n; ++i) dst[i] = fp16_to_fp32(p[i]);
            return 0;
        }
        case CHRIS_GGML_TYPE_Q4_0:
            if (n % 32) return -1;
            deq_q4_0((const block_q4_0 *)src, n/32, dst);
            return 0;
        case CHRIS_GGML_TYPE_Q4_1:
            if (n % 32) return -1;
            deq_q4_1((const block_q4_1 *)src, n/32, dst);
            return 0;
        case CHRIS_GGML_TYPE_Q5_0:
            if (n % 32) return -1;
            deq_q5_0((const block_q5_0 *)src, n/32, dst);
            return 0;
        case CHRIS_GGML_TYPE_Q5_1:
            if (n % 32) return -1;
            deq_q5_1((const block_q5_1 *)src, n/32, dst);
            return 0;
        case CHRIS_GGML_TYPE_Q8_0:
            if (n % 32) return -1;
            deq_q8_0((const block_q8_0 *)src, n/32, dst);
            return 0;
        case CHRIS_GGML_TYPE_Q4_K:
            if (n % QK_K) return -1;
            deq_q4_K((const block_q4_K *)src, n/QK_K, dst);
            return 0;
        case CHRIS_GGML_TYPE_Q5_K:
            if (n % QK_K) return -1;
            deq_q5_K((const block_q5_K *)src, n/QK_K, dst);
            return 0;
        case CHRIS_GGML_TYPE_Q6_K:
            if (n % QK_K) return -1;
            deq_q6_K((const block_q6_K *)src, n/QK_K, dst);
            return 0;
        default:
            return -1;
    }
}
