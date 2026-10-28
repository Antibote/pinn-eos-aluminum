/*
 * example.c — как расчётный код на C вызывает уравнение состояния PINN (без Python и PyTorch).
 *
 * Сборка и запуск (веса — из eos.cinfer.export_weights):
 *     gcc -O3 -march=native -ffast-math -fopenmp example.c eos_pinn.c -o example -lm
 *     ./example pinn.bin
 *
 * Печатает P, e в характерных точках и замеряет: время одного вызова eos_predict_one
 * (как в гидрокоде: по точке на ячейку) и пакет из 10^5 точек.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "eos_pinn.h"

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

int main(int argc, char **argv) {
    EOSPinn *m = eos_load(argc > 1 ? argv[1] : "pinn.bin");
    if (!m) return 1;
    printf("Параметров: %ld\n\n   rho, г/см3        T, K        P, ГПа      e, кДж/г\n", eos_n_params(m));
    const double pts[][2] = {{2.7, 300}, {10, 1e4}, {0.01, 1e6}, {1000, 1e5}, {1e-3, 0.1}};
    for (int i = 0; i < 5; i++) {
        double P, e;
        eos_predict_one(m, pts[i][0], pts[i][1], &P, &e);
        printf("%12.4g %12.4g %13.6g %13.6g\n", pts[i][0], pts[i][1], P, e);
    }

    const long n = 100000;
    double *rho = malloc(n * sizeof *rho), *T = malloc(n * sizeof *T);
    double *P = malloc(n * sizeof *P), *e = malloc(n * sizeof *e);
    srand(0);
    for (long i = 0; i < n; i++) { /* равномерно по log rho и log T во всей таблице */
        rho[i] = pow(10.0, -3.0 + 7.0 * rand() / RAND_MAX);
        T[i] = pow(10.0, -1.0 + 8.0 * rand() / RAND_MAX);
    }
    double sink = 0, t0 = now();
    for (long i = 0; i < n; i++) { /* по точке за вызов */
        double p, q;
        eos_predict_one(m, rho[i], T[i], &p, &q);
        sink += p;
    }
    double t1 = now();
    eos_predict(m, n, rho, T, P, e); /* прогрев потоков */
    double t2 = now();
    eos_predict(m, n, rho, T, P, e);
    double t3 = now();
    printf("\nОдин вызов eos_predict_one: %.2f мкс (10^5 вызовов за %.3f с)\n", (t1 - t0) / n * 1e6, t1 - t0);
    printf("Пакет eos_predict, 10^5 точек: %.3f с\n", t3 - t2);
    if (sink < 0) printf("%g\n", sink); /* чтобы компилятор не выбросил цикл */
    free(rho); free(T); free(P); free(e);
    eos_free(m);
    return 0;
}
