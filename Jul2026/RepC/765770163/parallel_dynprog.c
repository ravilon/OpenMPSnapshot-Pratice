#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include "../../PolybenchC/polybench.h"

/* Include benchmark-specific header. */
/* Default data type is int, default size is 50. */
#include "../../dynprog/dynprog.h"

#ifndef NTHREADS
#define NTHREADS 4
#endif

/* Array initialization. */
static void init_array(int length,
                       DATA_TYPE POLYBENCH_1D(c, LENGTH, length),
		                   DATA_TYPE POLYBENCH_1D(W, LENGTH, length))
{
  for (int i = 0; i < length; i++)
  {
    c[i] = 0;
	  W[i] = ((DATA_TYPE) -i) / length;
  }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(DATA_TYPE out)
{
  fprintf(stderr, DATA_PRINTF_MODIFIER, out);
  fprintf(stderr, "\n");
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_dynprog(int tsteps, int length,
                           DATA_TYPE POLYBENCH_1D(c, LENGTH, length),
                           DATA_TYPE POLYBENCH_1D(W, LENGTH, length),
                           DATA_TYPE sum_c,
                           DATA_TYPE *out)
{

  DATA_TYPE out_l = 0;
  sum_c = 0;

  for (int i = 1; i < _PB_LENGTH; i++)
  {
    #pragma omp parallel for num_threads(NTHREADS) reduction(+:sum_c)
    for (int j = 1; j < i; j++)
      sum_c += c[j];
    c[i] = sum_c + W[i];
    sum_c = 0;
  }

  for (int k = 0; k < _PB_TSTEPS; k++)
    out_l += c[_PB_LENGTH - 1];
  
  *out = out_l;

}

int main(int argc, char **argv)
{
  /* Retrieve problem size. */
  int length = LENGTH;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  DATA_TYPE out;
  DATA_TYPE sum_c;
  POLYBENCH_1D_ARRAY_DECL(c, DATA_TYPE, LENGTH, length);
  POLYBENCH_1D_ARRAY_DECL(W, DATA_TYPE, LENGTH, length);

  /* Initialize array(s). */
  init_array(length, POLYBENCH_ARRAY(c), POLYBENCH_ARRAY(W));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_dynprog(tsteps, length,
                 POLYBENCH_ARRAY(c),
                 POLYBENCH_ARRAY(W),
                 sum_c,
                 &out);

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(out));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(c);
  POLYBENCH_FREE_ARRAY(W);

  return 0;
}
