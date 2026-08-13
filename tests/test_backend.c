#include "../backend.h"

#include <math.h>
#include <stdio.h>

/*
 * Small backend smoke test. The expected values are calculated by hand so a
 * layout or reduction bug in the SYCL kernel is easy to spot.
 */
int main(void) {
    const float w[] = {
         1.0f,  2.0f,  3.0f,
        -1.0f,  0.5f,  4.0f
    };
    const float x[] = {2.0f, -1.0f, 0.5f};
    const float b[] = {0.25f, -0.75f};
    float y[2] = {0};

    ChrisBackend *backend = chris_backend_create("auto");
    if (!backend) return 1;
    if (chris_backend_preload(backend, w, 6) != 0 ||
        chris_backend_preload(backend, b, 2) != 0 ||
        chris_backend_matvec(backend, y, w, x, b, 3, 2) != 0) {
        chris_backend_destroy(backend);
        return 1;
    }

    /* y0 = 1*2 + 2*(-1) + 3*0.5 + 0.25 = 1.75
       y1 = -1*2 + 0.5*(-1) + 4*0.5 - 0.75 = -1.25 */
    if (fabsf(y[0] - 1.75f) > 1e-4f || fabsf(y[1] + 1.25f) > 1e-4f) {
        fprintf(stderr, "backend matvec mismatch: %.8f %.8f\n", y[0], y[1]);
        chris_backend_destroy(backend);
        return 1;
    }

    printf("backend=%s device=%s matvec OK\n",
           chris_backend_name(backend), chris_backend_device_name(backend));
    chris_backend_destroy(backend);
    return 0;
}
