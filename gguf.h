#ifndef CHRIS_GGUF_H
#define CHRIS_GGUF_H

#include "common.h"
#include "quant.h"

#define GGUF_MAGIC 0x46554747u  /* "GGUF" in little-endian. */

typedef enum {
    GGUF_VALUE_UINT8   = 0,
    GGUF_VALUE_INT8    = 1,
    GGUF_VALUE_UINT16  = 2,
    GGUF_VALUE_INT16   = 3,
    GGUF_VALUE_UINT32  = 4,
    GGUF_VALUE_INT32   = 5,
    GGUF_VALUE_FLOAT32 = 6,
    GGUF_VALUE_BOOL    = 7,
    GGUF_VALUE_STRING  = 8,
    GGUF_VALUE_ARRAY   = 9,
    GGUF_VALUE_UINT64  = 10,
    GGUF_VALUE_INT64   = 11,
    GGUF_VALUE_FLOAT64 = 12,
} GGUFValueType;

typedef struct {
    uint32_t elem_type;
    uint64_t count;
    void *data;  /* char ** for string arrays; packed bytes otherwise. */
} GGUFArray;

typedef struct {
    uint32_t type;
    union {
        uint64_t u64;
        int64_t i64;
        double f64;
        char *str;
        GGUFArray array;
    } as;
} GGUFValue;

typedef struct {
    char *key;
    GGUFValue value;
} GGUFMetadata;

typedef struct {
    char *name;
    uint32_t n_dims;
    uint64_t dims[4];  /* GGML order: dims[0] is contiguous. */
    uint32_t type;
    uint64_t offset;   /* Relative to the tensor data section. */
    uint64_t n_elements;
    size_t n_bytes;
} GGUFTensorInfo;

typedef struct {
    int fd;
    uint8_t *map;
    size_t file_size;

    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_metadata;
    GGUFMetadata *metadata;
    GGUFTensorInfo *tensors;

    uint64_t alignment;
    uint64_t data_offset;
} GGUFFile;

int gguf_open(const char *path, GGUFFile *gguf);
void gguf_close(GGUFFile *gguf);

const GGUFMetadata *gguf_find_metadata(const GGUFFile *gguf,
                                       const char *key);
const GGUFTensorInfo *gguf_find_tensor(const GGUFFile *gguf,
                                       const char *name);
const void *gguf_tensor_data(const GGUFFile *gguf,
                             const GGUFTensorInfo *tensor);

int gguf_get_u64(const GGUFFile *gguf, const char *key, uint64_t *out);
int gguf_get_f64(const GGUFFile *gguf, const char *key, double *out);
const char *gguf_get_string(const GGUFFile *gguf, const char *key);
const GGUFArray *gguf_get_array(const GGUFFile *gguf,
                                const char *key,
                                uint32_t element_type);

void gguf_print_summary(const GGUFFile *gguf, bool list_tensors);

#endif /* CHRIS_GGUF_H */
