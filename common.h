#ifndef CHRIS_COMMON_H
#define CHRIS_COMMON_H

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHRIS_UNUSED(x) ((void)(x))
#define CHRIS_MIN(a,b) ((a) < (b) ? (a) : (b))
#define CHRIS_MAX(a,b) ((a) > (b) ? (a) : (b))

static inline void *chris_xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "fatal: malloc(%zu) failed\n", n);
        exit(EXIT_FAILURE);
    }
    return p;
}

static inline void *chris_xcalloc(size_t n, size_t s) {
    void *p = calloc(n ? n : 1, s ? s : 1);
    if (!p) {
        fprintf(stderr, "fatal: calloc(%zu, %zu) failed\n", n, s);
        exit(EXIT_FAILURE);
    }
    return p;
}

static inline char *chris_xstrdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = (char *)chris_xmalloc(n);
    memcpy(p, s, n);
    return p;
}

#endif
