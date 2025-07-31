#include "walltime.h"
#include <math.h>
#include <stdlib.h>
#include <omp.h>

#define N_THR 8

int main(int argc, char *argv[])
{
  int N = 2000000000;
  double up = 1.00000001;
  double Sn = 1.00000001;

  omp_set_num_threads(N_THR);

  /* allocate memory for the recursion */
  double *opt = (double *)malloc((N + 1) * sizeof(double));

  if (opt == NULL)
    die("failed to allocate problem size");

  double time_start = wall_time();

  double Sn_local = Sn;
  int k = 0, n = 0;
#pragma omp parallel for default(none) firstprivate(Sn_local, k) lastprivate(Sn) shared(n, opt, up, N) schedule(guided, 150000)
  for (n = 0; n <= N; ++n)
  {
    if (k == 0 || k != n) {
      opt[n] = Sn_local * pow(up, n);
      Sn = opt[n] * up;
    } else {
      opt[n] = Sn;
      Sn *= up;
    }
  }

  printf("Parallel RunTime   :  %f seconds\n", wall_time() - time_start);
  printf("Final Result Sn    :  %.17g \n", Sn);

  double temp = 0.0;
  for (n = 0; n <= N; ++n)
  {
    temp += opt[n] * opt[n];
  }
  printf("Result ||opt||^2_2 :  %f\n", temp / (double)N);
  printf("\n");

  return 0;
}
