#pragma once

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

extern "C" {
#include "mmio.h"  // for input output
}

/**
* @brief Compressed Sparse Rows format of a Sparse matrix.
*
* @tparam T type of the values
*/
template <typename T>
struct CSR_Matrix {
int M, N, nz, *rowptr, *cols;
T *vals;
};

/**
* @brief Coordinate format for a Sparse matrix.
*/
typedef struct COO_Matrix {
int M, N, nz, *rows, *cols;
double *vals;
bool isSymmetric;
char type;  // 'r' = real, 'i' = integer, 'p' = pattern
} COO_Matrix;

/**
* @brief Free COO matrix in host memory.
*
* @param mat COO matrix
*/
void free_COO(COO_Matrix *mat);

/**
* @brief Allocate memory for a COO matrix in host memory.
*
* @param rows number of rows
* @param cols number of columns
* @param nonzeros number of non-zero values
* @return COO_Matrix*
*/
COO_Matrix *create_COO(int rows, int cols, int nonzeros);

/**
* @brief Read a COO matrix at the given path. The matrix should be from MatrixMarket.
*
* @param path path to MatrixMarket matrix
* @param isSymmetric this value will be updated to true if matrix is symetric
* @return COO_Matrix*
*/
COO_Matrix *read_COO(const char *path);

/**
* @brief Duplicates off-diagonal entries in a symmetric matrix, where just one triangle
* is given. Asserts the symmetry with isSymmetric ?= true.
*
* Implementation based on
* https://github.com/cusplibrary/cusplibrary/blob/develop/cusp/io/detail/matrix_market.inl#L244
*
* @param mat
*/
void duplicate_off_diagonals(COO_Matrix *mat);

/**
* @brief Change a matrix from COO format to CSR format.
*
* @param coo COO matrix
* @param compact defaults to false
* @return CSR_Matrix<T>*
*/
CSR_Matrix<double> *COO_to_CSR(COO_Matrix *coo);

// CSR Templates
template <typename T>
void free_CSR(CSR_Matrix<T> *mat);

template <typename T>
CSR_Matrix<T> *create_CSR(int rows, int cols, int nonzeros);

template <typename TSRC, typename TDEST>
CSR_Matrix<TDEST> *duplicate_CSR(CSR_Matrix<TSRC> *mat);

// import template function definitions
#include "matrix.tpp"