#include "../quant.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int check_all(const char *name, const float *v, size_t n, float expected) {
    for (size_t i = 0; i < n; ++i) {
        if (fabsf(v[i] - expected) > 1e-6f) {
            fprintf(stderr, "%s: index %zu = %.9g, expected %.9g\n",
                    name, i, v[i], expected);
            return -1;
        }
    }
    return 0;
}

int main(void) {
    float out[256];

    /* IEEE-754 half 1.0 = 0x3c00. */
    if (fabsf(fp16_to_fp32(0x3c00u) - 1.0f) > 1e-7f) return 1;

    /*
     * Q4_K: d=1, dmin=0, all eight sub-scales are 1 and every nibble is 8.
     * Every dequantized value should therefore be exactly 8.
     * Layout: d[2], dmin[2], scales[12], qs[128].
     */
    unsigned char q4k[144] = {0};
    q4k[0] = 0x00; q4k[1] = 0x3c; /* d = fp16(1) */
    /* Keep dmin at zero. */
    const unsigned char scales4[12] = {1,1,1,1,0,0,0,0,1,1,1,1};
    memcpy(q4k + 4, scales4, sizeof(scales4));
    memset(q4k + 16, 0x88, 128);
    if (quant_to_f32(CHRIS_GGML_TYPE_Q4_K, q4k, 256, out) != 0 ||
        check_all("Q4_K", out, 256, 8.0f) != 0) return 2;

    /*
     * Q5_K: low nibbles are 0 and all high bits are set, so q = 16.
     * With d=1, dmin=0, and sub-scales=1, every output value should be 16.
     * Layout: d[2], dmin[2], scales[12], qh[32], qs[128].
     */
    unsigned char q5k[176] = {0};
    q5k[0] = 0x00; q5k[1] = 0x3c;
    memcpy(q5k + 4, scales4, sizeof(scales4));
    memset(q5k + 16, 0xff, 32); /* high bit of every value */
    memset(q5k + 48, 0x00, 128);
    if (quant_to_f32(CHRIS_GGML_TYPE_Q5_K, q5k, 256, out) != 0 ||
        check_all("Q5_K", out, 256, 16.0f) != 0) return 3;

    /*
     * Q6_K: ql=0x11 and qh=0xaa encode q=33 in each of the four 6-bit groups.
     * The format is centered on 32, so scale=1 and d=1 gives 33-32 = 1.
     * Layout: ql[128], qh[64], scales[16], d[2].
     */
    unsigned char q6k[210] = {0};
    memset(q6k + 0, 0x11, 128);
    memset(q6k + 128, 0xaa, 64);
    memset(q6k + 192, 1, 16);
    q6k[208] = 0x00; q6k[209] = 0x3c;
    if (quant_to_f32(CHRIS_GGML_TYPE_Q6_K, q6k, 256, out) != 0 ||
        check_all("Q6_K", out, 256, 1.0f) != 0) return 4;

    puts("quant tests passed");
    return 0;
}
