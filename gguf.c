#include "gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    const uint8_t *cursor;
    const uint8_t *end;
} Reader;

static int read_bytes(Reader *reader, void *out, size_t count) {
    if ((size_t)(reader->end - reader->cursor) < count) {
        return -1;
    }

    if (out) {
        memcpy(out, reader->cursor, count);
    }
    reader->cursor += count;
    return 0;
}

static int read_u8(Reader *reader, uint8_t *value) {
    return read_bytes(reader, value, sizeof(*value));
}

static int read_i8(Reader *reader, int8_t *value) {
    return read_bytes(reader, value, sizeof(*value));
}

static int read_u16(Reader *reader, uint16_t *value) {
    return read_bytes(reader, value, sizeof(*value));
}

static int read_i16(Reader *reader, int16_t *value) {
    return read_bytes(reader, value, sizeof(*value));
}

static int read_u32(Reader *reader, uint32_t *value) {
    return read_bytes(reader, value, sizeof(*value));
}

static int read_i32(Reader *reader, int32_t *value) {
    return read_bytes(reader, value, sizeof(*value));
}

static int read_u64(Reader *reader, uint64_t *value) {
    return read_bytes(reader, value, sizeof(*value));
}

static int read_i64(Reader *reader, int64_t *value) {
    return read_bytes(reader, value, sizeof(*value));
}

static int read_f32(Reader *reader, float *value) {
    return read_bytes(reader, value, sizeof(*value));
}

static int read_f64(Reader *reader, double *value) {
    return read_bytes(reader, value, sizeof(*value));
}

static char *read_string(Reader *reader) {
    uint64_t length = 0;
    if (read_u64(reader, &length) != 0) {
        return NULL;
    }

    if (length > (uint64_t)(reader->end - reader->cursor) ||
        length > SIZE_MAX - 1) {
        return NULL;
    }

    char *text = (char *)chris_xmalloc((size_t)length + 1);
    if (read_bytes(reader, text, (size_t)length) != 0) {
        free(text);
        return NULL;
    }

    text[length] = '\0';
    return text;
}

static size_t scalar_size(uint32_t type) {
    switch (type) {
        case GGUF_VALUE_UINT8:
        case GGUF_VALUE_INT8:
        case GGUF_VALUE_BOOL:
            return 1;

        case GGUF_VALUE_UINT16:
        case GGUF_VALUE_INT16:
            return 2;

        case GGUF_VALUE_UINT32:
        case GGUF_VALUE_INT32:
        case GGUF_VALUE_FLOAT32:
            return 4;

        case GGUF_VALUE_UINT64:
        case GGUF_VALUE_INT64:
        case GGUF_VALUE_FLOAT64:
            return 8;

        default:
            return 0;
    }
}

static int parse_array(Reader *reader, GGUFArray *array) {
    uint32_t element_type = 0;
    uint64_t count = 0;

    if (read_u32(reader, &element_type) ||
        read_u64(reader, &count)) {
        return -1;
    }

    array->elem_type = element_type;
    array->count = count;

    if (element_type == GGUF_VALUE_STRING) {
        if (count > SIZE_MAX / sizeof(char *)) {
            return -1;
        }

        char **items = (char **)chris_xcalloc(
            (size_t)count,
            sizeof(char *));
        array->data = items;

        for (uint64_t i = 0; i < count; ++i) {
            items[i] = read_string(reader);
            if (!items[i]) {
                return -1;
            }
        }
        return 0;
    }

    const size_t element_size = scalar_size(element_type);
    if (!element_size || count > SIZE_MAX / element_size) {
        return -1;
    }

    const size_t bytes = (size_t)count * element_size;
    array->data = chris_xmalloc(bytes ? bytes : 1);
    return read_bytes(reader, array->data, bytes);
}

static int parse_value(Reader *reader,
                       uint32_t type,
                       GGUFValue *value) {
    memset(value, 0, sizeof(*value));
    value->type = type;

    switch (type) {
        case GGUF_VALUE_UINT8: {
            uint8_t item = 0;
            if (read_u8(reader, &item)) {
                return -1;
            }
            value->as.u64 = item;
            return 0;
        }

        case GGUF_VALUE_INT8: {
            int8_t item = 0;
            if (read_i8(reader, &item)) {
                return -1;
            }
            value->as.i64 = item;
            return 0;
        }

        case GGUF_VALUE_UINT16: {
            uint16_t item = 0;
            if (read_u16(reader, &item)) {
                return -1;
            }
            value->as.u64 = item;
            return 0;
        }

        case GGUF_VALUE_INT16: {
            int16_t item = 0;
            if (read_i16(reader, &item)) {
                return -1;
            }
            value->as.i64 = item;
            return 0;
        }

        case GGUF_VALUE_UINT32: {
            uint32_t item = 0;
            if (read_u32(reader, &item)) {
                return -1;
            }
            value->as.u64 = item;
            return 0;
        }

        case GGUF_VALUE_INT32: {
            int32_t item = 0;
            if (read_i32(reader, &item)) {
                return -1;
            }
            value->as.i64 = item;
            return 0;
        }

        case GGUF_VALUE_FLOAT32: {
            float item = 0.0f;
            if (read_f32(reader, &item)) {
                return -1;
            }
            value->as.f64 = item;
            return 0;
        }

        case GGUF_VALUE_BOOL: {
            uint8_t item = 0;
            if (read_u8(reader, &item)) {
                return -1;
            }
            value->as.u64 = item != 0;
            return 0;
        }

        case GGUF_VALUE_UINT64:
            return read_u64(reader, &value->as.u64);

        case GGUF_VALUE_INT64:
            return read_i64(reader, &value->as.i64);

        case GGUF_VALUE_FLOAT64:
            return read_f64(reader, &value->as.f64);

        case GGUF_VALUE_STRING:
            value->as.str = read_string(reader);
            return value->as.str ? 0 : -1;

        case GGUF_VALUE_ARRAY:
            return parse_array(reader, &value->as.array);

        default:
            return -1;
    }
}

static void free_value(GGUFValue *value) {
    if (!value) {
        return;
    }

    if (value->type == GGUF_VALUE_STRING) {
        free(value->as.str);
    } else if (value->type == GGUF_VALUE_ARRAY) {
        if (value->as.array.elem_type == GGUF_VALUE_STRING &&
            value->as.array.data) {
            char **items = (char **)value->as.array.data;
            for (uint64_t i = 0; i < value->as.array.count; ++i) {
                free(items[i]);
            }
        }
        free(value->as.array.data);
    }

    memset(value, 0, sizeof(*value));
}

const GGUFMetadata *gguf_find_metadata(const GGUFFile *gguf,
                                       const char *key) {
    if (!gguf || !key) {
        return NULL;
    }

    for (uint64_t i = 0; i < gguf->n_metadata; ++i) {
        if (strcmp(gguf->metadata[i].key, key) == 0) {
            return &gguf->metadata[i];
        }
    }

    return NULL;
}

const GGUFTensorInfo *gguf_find_tensor(const GGUFFile *gguf,
                                       const char *name) {
    if (!gguf || !name) {
        return NULL;
    }

    for (uint64_t i = 0; i < gguf->n_tensors; ++i) {
        if (strcmp(gguf->tensors[i].name, name) == 0) {
            return &gguf->tensors[i];
        }
    }

    return NULL;
}

static bool value_to_u64(const GGUFValue *value, uint64_t *out) {
    if (!value || !out) {
        return false;
    }

    switch (value->type) {
        case GGUF_VALUE_UINT8:
        case GGUF_VALUE_UINT16:
        case GGUF_VALUE_UINT32:
        case GGUF_VALUE_UINT64:
        case GGUF_VALUE_BOOL:
            *out = value->as.u64;
            return true;

        case GGUF_VALUE_INT8:
        case GGUF_VALUE_INT16:
        case GGUF_VALUE_INT32:
        case GGUF_VALUE_INT64:
            if (value->as.i64 < 0) {
                return false;
            }
            *out = (uint64_t)value->as.i64;
            return true;

        default:
            return false;
    }
}

int gguf_get_u64(const GGUFFile *gguf,
                 const char *key,
                 uint64_t *out) {
    const GGUFMetadata *metadata = gguf_find_metadata(gguf, key);
    return metadata && value_to_u64(&metadata->value, out) ? 0 : -1;
}

int gguf_get_f64(const GGUFFile *gguf,
                 const char *key,
                 double *out) {
    const GGUFMetadata *metadata = gguf_find_metadata(gguf, key);
    if (!metadata || !out) {
        return -1;
    }

    if (metadata->value.type == GGUF_VALUE_FLOAT32 ||
        metadata->value.type == GGUF_VALUE_FLOAT64) {
        *out = metadata->value.as.f64;
        return 0;
    }

    uint64_t integer = 0;
    if (value_to_u64(&metadata->value, &integer)) {
        *out = (double)integer;
        return 0;
    }

    return -1;
}

const char *gguf_get_string(const GGUFFile *gguf, const char *key) {
    const GGUFMetadata *metadata = gguf_find_metadata(gguf, key);
    if (!metadata || metadata->value.type != GGUF_VALUE_STRING) {
        return NULL;
    }
    return metadata->value.as.str;
}

const GGUFArray *gguf_get_array(const GGUFFile *gguf,
                                const char *key,
                                uint32_t element_type) {
    const GGUFMetadata *metadata = gguf_find_metadata(gguf, key);
    if (!metadata ||
        metadata->value.type != GGUF_VALUE_ARRAY ||
        metadata->value.as.array.elem_type != element_type) {
        return NULL;
    }
    return &metadata->value.as.array;
}

const void *gguf_tensor_data(const GGUFFile *gguf,
                             const GGUFTensorInfo *tensor) {
    if (!gguf || !tensor) {
        return NULL;
    }

    const uint64_t absolute_offset = gguf->data_offset + tensor->offset;
    if (absolute_offset > gguf->file_size ||
        tensor->n_bytes > gguf->file_size - absolute_offset) {
        return NULL;
    }

    return gguf->map + absolute_offset;
}

static int read_header(Reader *reader, GGUFFile *gguf) {
    uint32_t magic = 0;
    if (read_u32(reader, &magic) ||
        magic != GGUF_MAGIC ||
        read_u32(reader, &gguf->version)) {
        fprintf(stderr, "error: file is not a valid GGUF\n");
        return -1;
    }

    if (gguf->version < 2 || gguf->version > 3) {
        fprintf(stderr,
                "error: unsupported GGUF v%u (expected v2 or v3)\n",
                gguf->version);
        return -1;
    }

    if (read_u64(reader, &gguf->n_tensors) ||
        read_u64(reader, &gguf->n_metadata)) {
        return -1;
    }

    /* Reject clearly unreasonable counts before allocating parser structures. */
    if (gguf->n_tensors > 1000000 || gguf->n_metadata > 1000000) {
        return -1;
    }

    return 0;
}

static int read_metadata(Reader *reader, GGUFFile *gguf) {
    gguf->metadata = (GGUFMetadata *)chris_xcalloc(
        (size_t)gguf->n_metadata,
        sizeof(GGUFMetadata));

    for (uint64_t i = 0; i < gguf->n_metadata; ++i) {
        GGUFMetadata *metadata = &gguf->metadata[i];
        metadata->key = read_string(reader);

        uint32_t type = 0;
        if (!metadata->key ||
            read_u32(reader, &type) ||
            parse_value(reader, type, &metadata->value)) {
            fprintf(stderr,
                    "error: invalid GGUF metadata at index %" PRIu64 "\n",
                    i);
            return -1;
        }
    }

    uint64_t alignment = 0;
    if (gguf_get_u64(gguf, "general.alignment", &alignment) == 0 &&
        alignment > 0) {
        gguf->alignment = alignment;
    }

    if ((gguf->alignment & (gguf->alignment - 1)) != 0) {
        fprintf(stderr,
                "error: general.alignment must be a power of two\n");
        return -1;
    }

    return 0;
}

static int read_tensor_descriptors(Reader *reader, GGUFFile *gguf) {
    gguf->tensors = (GGUFTensorInfo *)chris_xcalloc(
        (size_t)gguf->n_tensors,
        sizeof(GGUFTensorInfo));

    for (uint64_t i = 0; i < gguf->n_tensors; ++i) {
        GGUFTensorInfo *tensor = &gguf->tensors[i];
        tensor->name = read_string(reader);

        if (!tensor->name ||
            read_u32(reader, &tensor->n_dims) ||
            tensor->n_dims == 0 ||
            tensor->n_dims > 4) {
            fprintf(stderr, "error: invalid GGUF tensor descriptor\n");
            return -1;
        }

        tensor->n_elements = 1;
        for (uint32_t dim = 0; dim < tensor->n_dims; ++dim) {
            if (read_u64(reader, &tensor->dims[dim]) ||
                tensor->dims[dim] == 0 ||
                tensor->n_elements > UINT64_MAX / tensor->dims[dim]) {
                return -1;
            }
            tensor->n_elements *= tensor->dims[dim];
        }

        if (read_u32(reader, &tensor->type) ||
            read_u64(reader, &tensor->offset)) {
            return -1;
        }

        /* Unknown types remain visible to --inspect. Model loading rejects
         * them only if it actually needs that tensor. */
        tensor->n_bytes = quant_nbytes(tensor->type,
                                       tensor->n_elements);
    }

    return 0;
}

static int validate_tensor_ranges(const GGUFFile *gguf) {
    for (uint64_t i = 0; i < gguf->n_tensors; ++i) {
        const GGUFTensorInfo *tensor = &gguf->tensors[i];
        if (!tensor->n_bytes) {
            continue;
        }

        const uint64_t absolute_offset =
            gguf->data_offset + tensor->offset;
        if (absolute_offset > gguf->file_size ||
            tensor->n_bytes > gguf->file_size - absolute_offset) {
            fprintf(stderr,
                    "error: tensor '%s' points outside the file\n",
                    tensor->name);
            return -1;
        }
    }

    return 0;
}

int gguf_open(const char *path, GGUFFile *gguf) {
    memset(gguf, 0, sizeof(*gguf));
    gguf->fd = -1;
    gguf->alignment = 32;

    gguf->fd = open(path, O_RDONLY);
    if (gguf->fd < 0) {
        perror("open GGUF");
        return -1;
    }

    struct stat stat_buffer;
    if (fstat(gguf->fd, &stat_buffer) != 0 || stat_buffer.st_size <= 0) {
        perror("fstat GGUF");
        gguf_close(gguf);
        return -1;
    }

    gguf->file_size = (size_t)stat_buffer.st_size;
    gguf->map = mmap(NULL,
                     gguf->file_size,
                     PROT_READ,
                     MAP_PRIVATE,
                     gguf->fd,
                     0);
    if (gguf->map == MAP_FAILED) {
        gguf->map = NULL;
        perror("mmap GGUF");
        gguf_close(gguf);
        return -1;
    }

    Reader reader = {
        .cursor = gguf->map,
        .end = gguf->map + gguf->file_size,
    };

    if (read_header(&reader, gguf) ||
        read_metadata(&reader, gguf) ||
        read_tensor_descriptors(&reader, gguf)) {
        gguf_close(gguf);
        return -1;
    }

    const uint64_t descriptor_end =
        (uint64_t)(reader.cursor - gguf->map);
    gguf->data_offset =
        (descriptor_end + gguf->alignment - 1) &
        ~(gguf->alignment - 1);

    if (gguf->data_offset > gguf->file_size ||
        validate_tensor_ranges(gguf)) {
        gguf_close(gguf);
        return -1;
    }

    return 0;
}

void gguf_close(GGUFFile *gguf) {
    if (!gguf) {
        return;
    }

    if (gguf->metadata) {
        for (uint64_t i = 0; i < gguf->n_metadata; ++i) {
            free(gguf->metadata[i].key);
            free_value(&gguf->metadata[i].value);
        }
        free(gguf->metadata);
    }

    if (gguf->tensors) {
        for (uint64_t i = 0; i < gguf->n_tensors; ++i) {
            free(gguf->tensors[i].name);
        }
        free(gguf->tensors);
    }

    if (gguf->map) {
        munmap(gguf->map, gguf->file_size);
    }
    if (gguf->fd >= 0) {
        close(gguf->fd);
    }

    memset(gguf, 0, sizeof(*gguf));
    gguf->fd = -1;
}

void gguf_print_summary(const GGUFFile *gguf, bool list_tensors) {
    printf("GGUF v%u | metadata=%" PRIu64
           " | tensors=%" PRIu64
           " | alignment=%" PRIu64 "\n",
           gguf->version,
           gguf->n_metadata,
           gguf->n_tensors,
           gguf->alignment);

    const char *architecture =
        gguf_get_string(gguf, "general.architecture");
    if (architecture) {
        printf("architecture: %s\n", architecture);
    }

    const char *name = gguf_get_string(gguf, "general.name");
    if (name) {
        printf("name: %s\n", name);
    }

    if (!list_tensors) {
        return;
    }

    for (uint64_t i = 0; i < gguf->n_tensors; ++i) {
        const GGUFTensorInfo *tensor = &gguf->tensors[i];
        printf("%-42s %-8s [",
               tensor->name,
               quant_type_name(tensor->type));

        for (uint32_t dim = 0; dim < tensor->n_dims; ++dim) {
            printf("%s%" PRIu64,
                   dim ? "," : "",
                   tensor->dims[dim]);
        }
        printf("]\n");
    }
}
