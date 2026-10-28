/*
 * eos_pinn.h — инференс обученной PINN уравнения состояния алюминия на чистом C.
 *
 * Только прямой проход: нормировка входов, слои Linear + активация, обратное
 * преобразование выходов. Обучение и производные (autograd) остаются в PyTorch.
 * Веса и нормировка читаются из бинарного файла, который пишет
 * eos.cinfer.export_weights (Python). Зависимостей, кроме libm (и OpenMP для
 * пакетного режима), нет — файл можно встроить в расчётный код на C/C++/Fortran.
 *
 * Пример:
 *     EOSPinn *m = eos_load("pinn.bin");
 *     double P, e;
 *     eos_predict_one(m, 2.7, 300.0, &P, &e);      // P [ГПа], e [кДж/г]
 *     eos_predict(m, n, rho, T, P_arr, e_arr);     // массив точек, параллельно
 *     eos_free(m);
 */
#ifndef EOS_PINN_H
#define EOS_PINN_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct EOSPinn EOSPinn;

/* Загружает модель из файла; NULL при ошибке (сообщение — в stderr). */
EOSPinn *eos_load(const char *path);

/* Освобождает память модели. */
void eos_free(EOSPinn *m);

/* Одна точка: rho [г/см³] > 0, T [K] > 0 → *P [ГПа], *e [кДж/г]. */
void eos_predict_one(const EOSPinn *m, double rho, double T, double *P, double *e);

/* n точек; считается блоками по 64 точки, блоки распределяются по потокам OpenMP. */
void eos_predict(const EOSPinn *m, long n, const double *rho, const double *T, double *P, double *e);

/* Число потоков OpenMP для eos_predict (0 — по умолчанию, все ядра). */
void eos_set_threads(int n);

/* Число параметров сети (для проверки, что загружена нужная модель). */
long eos_n_params(const EOSPinn *m);

#ifdef __cplusplus
}
#endif
#endif
