/*!****************************************************************************
* @file launcher-omp.c
* @brief Sample program to test MMCK performances (OpenMP version)
* This is a sample program that aims to a twofold purpose: to show a practical
* usage of the MMCK library (i.e., matrices multiplication), and to provide
* performance measuring for its functions.
*******************************************************************************/

#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include "../../omp/modules/io.h"
#include "../../omp/modules/computation.h"
#include "../../omp/modules/utils.h"
#include "../../omp/mmck.h"


/**
* Write execution statistics to an output file (in a @b sequential way),
* limited to the "preamble", i.e. all the statistics except for the average error,
* that will be computed (and appended to the stats file) externally via the
* the validator tool.
* @param np number of processes
* @param m M value
* @param k K value
* @param n N value
* @param time execution time
* @param avg_error average error
* @param pragma omp pragmas
*/
void
write_stats_preamble_omp(int np, int m, int k, int n, double time, int pragma) {
// Build the complete input file path
char *filnam = "stats-omp.csv";
char *complete_filename = MMCK_Utils_build_complete_filepath(filnam);

FILE *fh;
if ((fh = fopen(complete_filename, "a+")) == NULL) {
MMCK_Abort(0, MMCK_MODULES_IO, "Error opening stats file.");
}
double gflops = (double) ((2.0 * m * n * k) / time) * 1.0e-9;
if (fprintf(fh, "%d,%d,%d,%d,%d,%f,%f,",
m, k, n, np, pragma,
time,
gflops)
< 0) {
MMCK_Abort(0, MMCK_MODULES_IO, "Error writing to stats file.");
}

if (fclose(fh) != 0) {
MMCK_Abort(0, MMCK_MODULES_IO, "Error closing stats file.");
}
}

int main(int argc, char **argv) {
printf("Starting...\n");

if (argc != 9) {
printf("Usage: %s np m k n pragma filename_A filename_B filename_C\n"
"pragma: 0=none 1=private 2=shared 3=private+shared\n", argv[0]);
return 1;
}

int np = atoi(argv[1]);
int m = atoi(argv[2]);
int k = atoi(argv[3]);
int n = atoi(argv[4]);
int pragma = atoi(argv[5]);
char *matrixA_filename = argv[6];
char *matrixB_filename = argv[7];
char *matrixC_filename = argv[8];

// Start a canonical OpenMP program
omp_set_num_threads(np);

// ... do potentially whatever you want with OpenMP ...

// No ScaLAPACK integration is required for the OpenMP version, so each matrix is
// simply defined by a data array.
// Data arrays:
double *A = NULL, *B = NULL, *C = NULL;

/*=========================*
| Matrices data input     |
*-------------------------*/

double start = omp_get_wtime();
// Now load the generated matrices.
printf("Reading a %dx%d matrix from %s...\n", m, k, matrixA_filename);
MMCK_IO_parall_read_omp(matrixA_filename, m, k, &A);
//MMCK_Utils_print_matrix(A, m, k, "A");
printf("Reading a %dx%d matrix from %s...\n", k, n, matrixB_filename);
MMCK_IO_parall_read_omp(matrixB_filename, k, n, &B);
// MMCK_Utils_print_matrix(B, k, n, "B");
printf("Reading a %dx%d matrix from %s...\n", m, n, matrixC_filename);
MMCK_IO_parall_read_omp(matrixC_filename, m, n, &C);
// MMCK_Utils_print_matrix(C, m, n, "C");

/*=========================*
| Matrices multiplication |
*-------------------------*/

printf("Performing parallel multiplication...\n");
MMCK_Computation_parall_mult(m, n, k, A, B, C, pragma);

/*=========================*
| Matrices data output    |
*-------------------------*/

printf("Done. Writing the result...\n");
MMCK_IO_seq_write("RES-omp.dat", &C, m, n);
free(A);
free(B);
free(C);
double end = omp_get_wtime();

write_stats_preamble_omp(np,
m, k, n,
end - start, pragma);

printf("Terminating...\n");

return 0;
}



