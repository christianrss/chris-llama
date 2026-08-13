#ifndef CHRIS_BACKEND_H
#define CHRIS_BACKEND_H

#include <stddef.h>
#include <stdint.h>

/*
 * Compute backend used by the GPT-2 runtime.
 *
 * The model code calls this C interface and does not depend on SYCL types.
 * backend_cpu.c implements the reference path; backend_acpp.cpp implements the
 * AdaptiveCpp path behind the same ABI.
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ChrisBackend ChrisBackend;

/*
 * Create a backend for the requested device.
 *
 * device_spec may be:
 *   "auto"  - prefer a GPU, otherwise use the first available device
 *   "gpu"   - require a GPU
 *   "cpu"   - require a CPU device
 *   "N"     - select the index printed by --list-devices
 *   text    - case-insensitive substring of the device name
 */
ChrisBackend *chris_backend_create(const char *device_spec);
void chris_backend_destroy(ChrisBackend *backend);

void chris_backend_list_devices(void);

const char *chris_backend_name(const ChrisBackend *backend);
const char *chris_backend_device_name(const ChrisBackend *backend);
int chris_backend_is_accelerated(const ChrisBackend *backend);

/*
 * Make a read-only host buffer available to the backend before generation.
 * AdaptiveCpp keeps a device copy until the backend is destroyed. The CPU
 * backend has nothing to preload and returns immediately.
 */
int chris_backend_preload(ChrisBackend *backend,
                          const float *host_data,
                          size_t element_count);

/*
 * Compute y = W*x + bias.
 *
 * The GGUF loader exposes matrices as row-major FP32 data with shape
 * [out_features][in_features]. The AdaptiveCpp backend keeps W and bias on the
 * device and transfers only x and y for each call.
 */
int chris_backend_matvec(ChrisBackend *backend,
                         float *out,
                         const float *weights,
                         const float *x,
                         const float *bias,
                         uint32_t in_features,
                         uint32_t out_features);

#ifdef __cplusplus
}
#endif

#endif /* CHRIS_BACKEND_H */
