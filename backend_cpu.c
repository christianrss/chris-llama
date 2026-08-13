#include "backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Plain C reference backend.
 *
 * This path is intentionally simple. It keeps the runtime usable without
 * AdaptiveCpp and gives us a deterministic result to compare against a SYCL
 * device when bringing up a new backend.
 */
struct ChrisBackend {
    char name[32];
};

ChrisBackend *chris_backend_create(const char *device_spec) {
    if (device_spec &&
        strcmp(device_spec, "auto") != 0 &&
        strcmp(device_spec, "cpu") != 0 &&
        strcmp(device_spec, "0") != 0) {
        fprintf(stderr,
                "error: the CPU backend only exposes device 'cpu' (index 0); "
                "received '%s'\n",
                device_spec);
        return NULL;
    }

    ChrisBackend *backend = (ChrisBackend *)calloc(1, sizeof(*backend));
    if (!backend) {
        return NULL;
    }

    strcpy(backend->name, "CPU reference");
    return backend;
}

void chris_backend_destroy(ChrisBackend *backend) {
    free(backend);
}

void chris_backend_list_devices(void) {
    printf("[0] CPU reference (serial C)\n");
}

const char *chris_backend_name(const ChrisBackend *backend) {
    (void)backend;
    return "cpu";
}

const char *chris_backend_device_name(const ChrisBackend *backend) {
    return backend ? backend->name : "<none>";
}

int chris_backend_is_accelerated(const ChrisBackend *backend) {
    (void)backend;
    return 0;
}

int chris_backend_preload(ChrisBackend *backend,
                          const float *host_data,
                          size_t element_count) {
    (void)backend;
    (void)host_data;
    (void)element_count;
    return 0;
}

int chris_backend_matvec(ChrisBackend *backend,
                         float *out,
                         const float *weights,
                         const float *x,
                         const float *bias,
                         uint32_t in_features,
                         uint32_t out_features) {
    if (!backend || !out || !weights || !x || !in_features || !out_features) {
        return -1;
    }

    for (uint32_t row_index = 0; row_index < out_features; ++row_index) {
        const float *row = weights + (size_t)row_index * in_features;
        float sum = bias ? bias[row_index] : 0.0f;

        for (uint32_t i = 0; i < in_features; ++i) {
            sum += row[i] * x[i];
        }

        out[row_index] = sum;
    }

    return 0;
}
