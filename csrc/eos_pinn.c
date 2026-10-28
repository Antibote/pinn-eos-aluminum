/*
 * eos_pinn.c — прямой проход PINN (см. eos_pinn.h).
 *
 * Формат файла весов (little-endian), его пишет eos.cinfer.export_weights:
 *   char[4] "EOSP"; int32 версия = 1; int32 активация (1 GELU, 2 SiLU, 3 tanh); int32 число слоёв;
 *   double x_min[2], x_max[2], y_center[2], y_scale[2], E0;
 *   для каждого слоя: int32 in, out, есть_смещение, активация_после;
 *                     float W[out][in]; float b[out] (если есть смещение).
 * Слои без активации между ними — это SVD-сжатые пары 256→r→256.
 */
#include "eos_pinn.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define BLOCK 64 /* точек в блоке: веса слоя переиспользуются для всех точек блока из кэша */

typedef struct {
    int in, out, act_after;
    float *W, *b; /* W[out][in], b[out] или NULL */
} Layer;

struct EOSPinn {
    int act, n_layers, max_dim;
    double x_min[2], x_max[2], y_center[2], y_scale[2], E0;
    Layer *layers;
};

/* erf(x) по Абрамовицу—Стигану 7.1.26: |ошибка| < 1,5e-7 — на уровне точности float32.
 * Без ветвлений, через expf — цикл активации векторизуется (libmvec), в отличие от erff. */
static inline float erf_fast(float x) {
    float s = x < 0.0f ? -1.0f : 1.0f, a = fabsf(x);
    float t = 1.0f / (1.0f + 0.3275911f * a);
    float y = ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
    return s * (1.0f - y * expf(-a * a));
}

/* Активация на месте для вектора длины n. */
static void activate(int act, float *v, int n) {
    switch (act) {
    case 1: for (int i = 0; i < n; i++) v[i] = 0.5f * v[i] * (1.0f + erf_fast(v[i] * 0.70710678118654752f)); break; /* GELU */
    case 2: for (int i = 0; i < n; i++) v[i] = v[i] / (1.0f + expf(-v[i])); break;                                    /* SiLU */
    default: for (int i = 0; i < n; i++) v[i] = tanhf(v[i]);
    }
}

static int read_all(FILE *f, void *dst, size_t size, size_t n) { return fread(dst, size, n, f) == n; }

EOSPinn *eos_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "eos_load: не открыть %s\n", path); return NULL; }
    char magic[4]; int32_t hdr[3];
    EOSPinn *m = calloc(1, sizeof *m);
    if (!read_all(f, magic, 1, 4) || memcmp(magic, "EOSP", 4) || !read_all(f, hdr, 4, 3) || hdr[0] != 1) {
        fprintf(stderr, "eos_load: %s — не файл весов EOSP v1\n", path); goto fail;
    }
    m->act = hdr[1]; m->n_layers = hdr[2]; m->max_dim = 2;
    double norm[9];
    if (!read_all(f, norm, 8, 9)) goto bad;
    memcpy(m->x_min, norm, 2 * 8); memcpy(m->x_max, norm + 2, 2 * 8);
    memcpy(m->y_center, norm + 4, 2 * 8); memcpy(m->y_scale, norm + 6, 2 * 8); m->E0 = norm[8];
    m->layers = calloc(m->n_layers, sizeof(Layer));
    for (int k = 0; k < m->n_layers; k++) {
        int32_t d[4]; Layer *L = &m->layers[k];
        if (!read_all(f, d, 4, 4)) goto bad;
        L->in = d[0]; L->out = d[1]; L->act_after = d[3];
        L->W = malloc(sizeof(float) * L->in * L->out);
        if (!read_all(f, L->W, 4, (size_t)L->in * L->out)) goto bad;
        if (d[2]) { L->b = malloc(sizeof(float) * L->out); if (!read_all(f, L->b, 4, L->out)) goto bad; }
        if (L->out > m->max_dim) m->max_dim = L->out;
    }
    fclose(f);
    return m;
bad:
    fprintf(stderr, "eos_load: %s обрезан\n", path);
fail:
    fclose(f); eos_free(m); return NULL;
}

void eos_free(EOSPinn *m) {
    if (!m) return;
    for (int k = 0; m->layers && k < m->n_layers; k++) { free(m->layers[k].W); free(m->layers[k].b); }
    free(m->layers); free(m);
}

void eos_set_threads(int n) {
#ifdef _OPENMP
    omp_set_num_threads(n > 0 ? n : omp_get_num_procs());
#else
    (void)n;
#endif
}

long eos_n_params(const EOSPinn *m) {
    long n = 0;
    for (int k = 0; k < m->n_layers; k++) n += (long)m->layers[k].in * m->layers[k].out + (m->layers[k].b ? m->layers[k].out : 0);
    return n;
}

/* Прямой проход для блока из nb ≤ BLOCK точек; buf_a, buf_b — рабочие буферы BLOCK × max_dim. */
static void forward_block(const EOSPinn *m, int nb, const double *rho, const double *T,
                          double *P, double *e, float *buf_a, float *buf_b) {
    float *x = buf_a, *y = buf_b;
    for (int p = 0; p < nb; p++) { /* log10 и перевод в [-1, 1] — как EOSNormalizer.torch_x */
        x[2 * p]     = (float)(2.0 * (log10(rho[p]) - m->x_min[0]) / (m->x_max[0] - m->x_min[0]) - 1.0);
        x[2 * p + 1] = (float)(2.0 * (log10(T[p])   - m->x_min[1]) / (m->x_max[1] - m->x_min[1]) - 1.0);
    }
    for (int k = 0; k < m->n_layers; k++) {
        const Layer *L = &m->layers[k];
        int p = 0;
        for (; p + 4 <= nb; p += 4) { /* 4 точки на строку весов: W читается вчетверо реже, 4 независимых накопителя */
            const float *x0 = x + (size_t)p * L->in, *x1 = x0 + L->in, *x2 = x1 + L->in, *x3 = x2 + L->in;
            float *y0 = y + (size_t)p * L->out, *y1 = y0 + L->out, *y2 = y1 + L->out, *y3 = y2 + L->out;
            for (int j = 0; j < L->out; j++) {
                const float *w = L->W + (size_t)j * L->in;
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                for (int i = 0; i < L->in; i++) { /* векторизуется: -O3 -march=native -ffast-math */
                    float wi = w[i];
                    a0 += wi * x0[i]; a1 += wi * x1[i]; a2 += wi * x2[i]; a3 += wi * x3[i];
                }
                float bj = L->b ? L->b[j] : 0.0f;
                y0[j] = a0 + bj; y1[j] = a1 + bj; y2[j] = a2 + bj; y3[j] = a3 + bj;
            }
        }
        for (; p < nb; p++) { /* остаток блока (и режим одной точки) */
            const float *xp = x + (size_t)p * L->in;
            float *yp = y + (size_t)p * L->out;
            for (int j = 0; j < L->out; j++) {
                const float *w = L->W + (size_t)j * L->in;
                float acc = 0.0f;
                for (int i = 0; i < L->in; i++) acc += w[i] * xp[i];
                yp[j] = L->b ? acc + L->b[j] : acc;
            }
        }
        if (L->act_after) activate(m->act, y, nb * L->out);
        float *t = x; x = y; y = t;
    }
    for (int p = 0; p < nb; p++) { /* обратная нормировка — как EOSNormalizer.torch_inverse_y */
        double lp = x[2 * p] * m->y_scale[0] + m->y_center[0];
        double ae = x[2 * p + 1] * m->y_scale[1] + m->y_center[1];
        P[p] = pow(10.0, lp);
        e[p] = m->E0 * sinh(ae);
    }
}

void eos_predict_one(const EOSPinn *m, double rho, double T, double *P, double *e) {
    float a[1024], b[1024]; /* одна точка: буферы на стеке (сеть шириной до 1024) */
    forward_block(m, 1, &rho, &T, P, e, a, b);
}

void eos_predict(const EOSPinn *m, long n, const double *rho, const double *T, double *P, double *e) {
    long n_blocks = (n + BLOCK - 1) / BLOCK;
#pragma omp parallel
    {
        float *a = malloc(sizeof(float) * BLOCK * m->max_dim);
        float *b = malloc(sizeof(float) * BLOCK * m->max_dim);
#pragma omp for schedule(static)
        for (long blk = 0; blk < n_blocks; blk++) {
            long s = blk * BLOCK;
            int nb = (int)(n - s < BLOCK ? n - s : BLOCK);
            forward_block(m, nb, rho + s, T + s, P + s, e + s, a, b);
        }
        free(a); free(b);
    }
}
