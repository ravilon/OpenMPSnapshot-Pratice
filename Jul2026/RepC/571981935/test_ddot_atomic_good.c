#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Compute inner product:
 *    s = x[0]*y[0] + x[1]*y[1] + ... + x[n]*y[n]
 * where x and y are double-precision n-vectors.
 * NOTE: This code uses private variables and atomics.
 */
int main() {
  /* Shared variables. */
  size_t n = 10000000;
  double *x = malloc(n * sizeof *x);
  double *y = malloc(n * sizeof *y);
  double s = 0;
  double reference_solution = 0;
  double tmp;

  #pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    x[i] = 1.23 * i;
    y[i] = 4.56 * i;
  }

  /* Sum of first n squares is n * (n+1) * (2n+1) / 6. */
  reference_solution = 1.23 * 4.56 * (n-1) * n * (2*n-1) / 6;

  #pragma omp parallel private (tmp)
  {
    tmp = 0;
    #pragma omp for
    for (int i = 0; i < n; i++)
      tmp += x[i] * y[i];
    #pragma omp atomic update
    s += tmp;
  }
  printf("Expected result: %.5e\nComputed result: %.5e\n",
	 reference_solution, s);

  free(x);
  free(y);
}
