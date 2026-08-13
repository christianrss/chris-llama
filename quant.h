#ifndef CHRIS_QUANT_H
#define CHRIS_QUANT_H

#include "common.h"

/* ggml_type IDs used by GGUF. */
enum {
    CHRIS_GGML_TYPE_F32  = 0,
    CHRIS_GGML_TYPE_F16  = 1,
    CHRIS_GGML_TYPE_Q4_0 = 2,
    CHRIS_GGML_TYPE_Q4_1 = 3,
    CHRIS_GGML_TYPE_Q5_0 = 6,
    CHRIS_GGML_TYPE_Q5_1 = 7,
    CHRIS_GGML_TYPE_Q8_0 = 8,
    CHRIS_GGML_TYPE_Q4_K = 12,
    CHRIS_GGML_TYPE_Q5_K = 13,
    CHRIS_GGML_TYPE_Q6_K = 14,
};

size_t quant_nbytes(uint32_t type, uint64_t element_count);
int quant_to_f32(uint32_t type,
                 const void *source,
                 uint64_t element_count,
                 float *destination);

float fp16_to_fp32(uint16_t value);
const char *quant_type_name(uint32_t type);

#endif /* CHRIS_QUANT_H */
