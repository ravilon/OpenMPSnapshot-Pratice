#include "lu/openmp.h"

#include <math.h>
#include <omp.h>

#include "common.h"

LU_t omp_LU(matrix_t* matrix) {
int m = (int) matrix->m;
int n = (int) matrix->n;
double* l = malloc(sizeof(double) * n * n);
double* u = malloc(sizeof(double) * n * n);
double* A = matrix_to_array(matrix);
int i, j, z, k = 0, flag = 0;
for (i = 0; i < n; i++) {
array_set(l, n, i + 1, i + 1, 1.0);
double sum_u = 0.0, sum_l = 0.0, val = 0.0, divd;

#pragma omp parallel for reduction(+:sum_u) firstprivate(val, n, i) private(k)
for (k = 0; k < i; k++) {
sum_u += array_get(u, n, k + 1, i + 1) * array_get(l, n, i + 1, k + 1);
}
array_set(u, n, i + 1, i + 1, (array_get(A, n, i + 1, i + 1) - sum_u));

#pragma omp parallel for firstprivate(n, i) private(sum_u, sum_l, k, divd, m)
for (j = 0; j < 2 * (n - i - 1); j++) {
if (j < n - i - 1) {
sum_u = 0.0;
for (k = 0; k < i; k++) {
sum_u += array_get(u, n, k + 1, j + i + 2) * array_get(l, n, i + 1, k + 1);
}
array_set(u, n, i + 1, j + i + 2, (array_get(A, n, i + 1, j + i + 2) - sum_u));
} else {
z = j % (n - i - 1);
sum_l = 0.0;
for (k = 0; k < i;  k++) {
sum_l += array_get(u, n, k + 1, i + 1) * array_get(l, n, z + i + 2, k + 1);
}
divd = array_get(u, n, i + 1, i + 1);
if (divd == 0.0) flag = -1;
array_set(l, n, z + i + 2, i + 1, (array_get(A, n, z + i + 2, i + 1) - sum_l) / divd);
}
}
}
// Выдача результата
LU_t result;
result.L = matrix_create((size_t) m, (size_t) n, l);
result.U = matrix_create((size_t) m, (size_t) n, u);
// Очистка промежуточных значений
free(l);
free(u);
free(A);
return result;
}
