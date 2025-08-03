/*!****************************************************************************
* @file computation.c
* @brief Code implementation for MMCK-OMP "Computation" module.
* This file contains implementation for what is defined in the "I/O"
* module header file.
*******************************************************************************/

#include "../computation.h"
#include "../../mmck.h"

void MMCK_Computation_parall_mult(int m, int n, int k, const double *A, const double *B, double *C, int pragmas) {
int i_tile, k_tile, j_tile;
int i_mat, k_mat, j_mat;

// NOTICE: code replication is necessary for the correct functioning of the macros
switch (pragmas) {
case MMCK_COMPUTATION_PRAGMAS_NONE:
#pragma omp parallel for
for (i_tile = 0; i_tile < m; i_tile += MMCK_COMPUTATION_TILE_SIZE) {
int imax = i_tile + MMCK_COMPUTATION_TILE_SIZE > m ? m : i_tile + MMCK_COMPUTATION_TILE_SIZE;
for (k_tile = 0; k_tile < k; k_tile += MMCK_COMPUTATION_TILE_SIZE) {
int kmax = k_tile + MMCK_COMPUTATION_TILE_SIZE > k ? k : k_tile + MMCK_COMPUTATION_TILE_SIZE;
for (j_tile = 0; j_tile < n; j_tile += MMCK_COMPUTATION_TILE_SIZE) {
int jmax = j_tile + MMCK_COMPUTATION_TILE_SIZE > n ? n : j_tile + MMCK_COMPUTATION_TILE_SIZE;
for (i_mat = i_tile; i_mat < imax; ++i_mat) {
for (k_mat = k_tile; k_mat < kmax; ++k_mat) {
for (j_mat = j_tile; j_mat < jmax; ++j_mat) {
C[MMCK_COORDS_TO_LINEAR_2D(i_mat, j_mat, m, n)] +=
A[MMCK_COORDS_TO_LINEAR_2D(i_mat, k_mat, m, k)] *
B[MMCK_COORDS_TO_LINEAR_2D(k_mat, j_mat, k, n)];
}
}
}
}
}
}
break;
case MMCK_COMPUTATION_PRAGMAS_PRIVATE:
#pragma omp parallel for private(i_tile, j_tile, k_tile, i_mat, j_mat, k_mat)
for (i_tile = 0; i_tile < m; i_tile += MMCK_COMPUTATION_TILE_SIZE) {
int imax = i_tile + MMCK_COMPUTATION_TILE_SIZE > m ? m : i_tile + MMCK_COMPUTATION_TILE_SIZE;
for (k_tile = 0; k_tile < k; k_tile += MMCK_COMPUTATION_TILE_SIZE) {
int kmax = k_tile + MMCK_COMPUTATION_TILE_SIZE > k ? k : k_tile + MMCK_COMPUTATION_TILE_SIZE;
for (j_tile = 0; j_tile < n; j_tile += MMCK_COMPUTATION_TILE_SIZE) {
int jmax = j_tile + MMCK_COMPUTATION_TILE_SIZE > n ? n : j_tile + MMCK_COMPUTATION_TILE_SIZE;
for (i_mat = i_tile; i_mat < imax; ++i_mat) {
for (k_mat = k_tile; k_mat < kmax; ++k_mat) {
for (j_mat = j_tile; j_mat < jmax; ++j_mat) {
C[MMCK_COORDS_TO_LINEAR_2D(i_mat, j_mat, m, n)] +=
A[MMCK_COORDS_TO_LINEAR_2D(i_mat, k_mat, m, k)] *
B[MMCK_COORDS_TO_LINEAR_2D(k_mat, j_mat, k, n)];
}
}
}
}
}
}
break;
case MMCK_COMPUTATION_PRAGMAS_SHARED:
#pragma omp parallel for shared(A, B, C)
for (i_tile = 0; i_tile < m; i_tile += MMCK_COMPUTATION_TILE_SIZE) {
int imax = i_tile + MMCK_COMPUTATION_TILE_SIZE > m ? m : i_tile + MMCK_COMPUTATION_TILE_SIZE;
for (k_tile = 0; k_tile < k; k_tile += MMCK_COMPUTATION_TILE_SIZE) {
int kmax = k_tile + MMCK_COMPUTATION_TILE_SIZE > k ? k : k_tile + MMCK_COMPUTATION_TILE_SIZE;
for (j_tile = 0; j_tile < n; j_tile += MMCK_COMPUTATION_TILE_SIZE) {
int jmax = j_tile + MMCK_COMPUTATION_TILE_SIZE > n ? n : j_tile + MMCK_COMPUTATION_TILE_SIZE;
for (i_mat = i_tile; i_mat < imax; ++i_mat) {
for (k_mat = k_tile; k_mat < kmax; ++k_mat) {
for (j_mat = j_tile; j_mat < jmax; ++j_mat) {
C[MMCK_COORDS_TO_LINEAR_2D(i_mat, j_mat, m, n)] +=
A[MMCK_COORDS_TO_LINEAR_2D(i_mat, k_mat, m, k)] *
B[MMCK_COORDS_TO_LINEAR_2D(k_mat, j_mat, k, n)];
}
}
}
}
}
}
break;
case MMCK_COMPUTATION_PRAGMAS_ALL:
#pragma omp parallel for private(i_tile, j_tile, k_tile, i_mat, j_mat, k_mat) shared(A, B, C)
for (i_tile = 0; i_tile < m; i_tile += MMCK_COMPUTATION_TILE_SIZE) {
int imax = i_tile + MMCK_COMPUTATION_TILE_SIZE > m ? m : i_tile + MMCK_COMPUTATION_TILE_SIZE;
for (k_tile = 0; k_tile < k; k_tile += MMCK_COMPUTATION_TILE_SIZE) {
int kmax = k_tile + MMCK_COMPUTATION_TILE_SIZE > k ? k : k_tile + MMCK_COMPUTATION_TILE_SIZE;
for (j_tile = 0; j_tile < n; j_tile += MMCK_COMPUTATION_TILE_SIZE) {
int jmax = j_tile + MMCK_COMPUTATION_TILE_SIZE > n ? n : j_tile + MMCK_COMPUTATION_TILE_SIZE;
for (i_mat = i_tile; i_mat < imax; ++i_mat) {
for (k_mat = k_tile; k_mat < kmax; ++k_mat) {
for (j_mat = j_tile; j_mat < jmax; ++j_mat) {
C[MMCK_COORDS_TO_LINEAR_2D(i_mat, j_mat, m, n)] +=
A[MMCK_COORDS_TO_LINEAR_2D(i_mat, k_mat, m, k)] *
B[MMCK_COORDS_TO_LINEAR_2D(k_mat, j_mat, k, n)];
}
}
}
}
}
}
break;
default:
MMCK_Abort(0, MMCK_MODULES_COMPUTATION, "Invalid pragma provided.");
break;
}
}

void MMCK_Computation_seq_mult(int m, int n, int k, double *A, double *B, double *C) {
for (int i = 0; i < m; ++i) {
for (int kk = 0; kk < k; ++kk) {
for (int j = 0; j < n; ++j) {
C[MMCK_COORDS_TO_LINEAR_2D(i, j, m, n)] +=
A[MMCK_COORDS_TO_LINEAR_2D(i, kk, m, k)] *
B[MMCK_COORDS_TO_LINEAR_2D(kk, j, k, n)];
}
}
}
}
