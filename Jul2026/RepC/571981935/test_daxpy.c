#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Compute daxpy:
 *     z <- alpha*x + y
 * where z, x, y are double-precision n-vectors and
 * alpha is a double-precision scalar.
 */

/* If the macro USE_EXPENSIVE_LOOP is defined at compile time, the operation
 *     z[i] = alpha * x[i] + y[i];
 * is replaced by the mathematically equivalent—but computationally more
 * expensive—formulation
 *     z[i] = log(exp(alpha * x[i]) * exp(y[i]));
 * Note that the logarithm and exponential functions require include in the C
 * math library header and linking against the corresponding library (hence the
 * -lm flag in the Makefile).
 */

#ifdef USE_EXPENSIVE_LOOP
#include <math.h>
#endif /* #ifdef USE_EXPENSIVE_LOOP */

int main() {

  /* Shared variables. */
  size_t n = 100000000;
  double *x = malloc(n * sizeof *x);
  double *y = malloc(n * sizeof *y);
  double *z = malloc(n * sizeof *z);
  double alpha = 2.23;

  double start_time = omp_get_wtime();
  #pragma omp parallel
  {
    #pragma omp for
    for (int i = 0; i < n; ++i) {
      x[i] = 1.23 * i;
      y[i] = 4.56 * i;
    }

    /* This barrier is unnecessary here. */
    #pragma omp barrier

    #pragma omp for
    for (int i = 0; i < n; ++i) {
#ifndef USE_EXPENSIVE_LOOP
      z[i] = alpha * x[i] + y[i];
#else
      z[i] = log(exp(alpha * x[i]) * exp(y[i]));
#endif /* #ifndef EXPENSIVE_LOOP */
    }
  }
  double time = omp_get_wtime() - start_time;
  printf("Max threads:  %3d\n", omp_get_max_threads());
  printf("Elapsed time: %.3es\n", time);

  free(x);
  free(y);
  free(z);
}
