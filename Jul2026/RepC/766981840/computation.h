/*!****************************************************************************
 * @file computation.h
 * @brief Main header file for MMCK-OMP "Computation" module.
 * This header file contains macros and prototypes for all the purposes of
 * data mathematical computation.
 *******************************************************************************/

#ifndef OMP_COMPUTATION_H
#define OMP_COMPUTATION_H

/**
 * Size of the "tile" when tiling (machine-dependent)
 */
#define MMCK_COMPUTATION_TILE_SIZE 128

/*=========================*
 | Pragmas                 |
 *-------------------------*/

/**
 * Do not use any additional #pragma
 */
#define MMCK_COMPUTATION_PRAGMAS_NONE       0

/**
 * Use private(i, j, kk) #pragma
 */
#define MMCK_COMPUTATION_PRAGMAS_PRIVATE    1

/**
 * Use shared(A, B, C) #pragma
 */
#define MMCK_COMPUTATION_PRAGMAS_SHARED     2

/**
 * Use private(i, j, kk) shared(A, B, C) #pragmas
 */
#define MMCK_COMPUTATION_PRAGMAS_ALL        3

/*=========================*
 | Parallel operations     |
 *-------------------------*/

/**
 * Perform a matrices @b multiplication C += A * B in a @b parallel way.
 * Internal matrices dimensions checks are provided.
 * The final output will be put into the @c C parameter
 * @param m value of M
 * @param n value of N
 * @param k value of K
 * @param A matrix A data array
 * @param B matrix B data array
 * @param C matrix C data array - will also contain the final output
 * @param pragmas any of @c MMCK_COMPUTATION_PRAGMAS_xx values
 */
void MMCK_Computation_parall_mult(int m, int n, int k, const double *A, const double *B, double *C, int pragmas);

/*=========================*
 | Sequential operations   |
 *-------------------------*/

/**
 *
 * Perform a matrices @b multiplication C += A * B in a @b sequential way.
 * The final output will be put into the @c C parameter
 * @param m value of M
 * @param n value of N
 * @param k value of K
 * @param A matrix A data array
 * @param B matrix B data array
 * @param C matrix C data array - will also contain the final output
 */
void MMCK_Computation_seq_mult(int m, int n, int k, double *A, double *B, double *C);

#endif //OMP_COMPUTATION_H
