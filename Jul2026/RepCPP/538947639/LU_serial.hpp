#pragma once

#include "matrix_span_serial.hpp"


// # LU (serial) #
// - No pivoting
// - Time complexity O(2/3 N^3)
//
// >>> Method goes over rows with following iterations:
//
// 1) 'Normalize' i-th row
//
//   0 0 0 0 0 0 0 0   (zeroes represent already 'handled' elements) 
//   0 0 0 0 0 0 0 0
//   0 0 M # # # # #   <- take inverse of M and multiply all subsequent elements in a row by it
//   0 0 v v v v v v
//   0 0 v v v v v v   ('#' represents elements that are changed on a current step)
//   0 0 v v v v v v
//   0 0 v v v v v v
//   0 0 v v v v v v
//
// 2) Transform bottom-right block
//
//   0 0 0 0 0 0 0 0
//   0 0 0 0 0 0 0 0
//   0 0 M - U - - -   
//   0 0 L v # v v v   <- go over all elements 'v' and substract 'L * U'
//   0 0 v v v v v v      to get a new value '#'
//   0 0 v v v v v v
//   0 0 v v v v v v      
//   0 0 v v v v v v
//
// After completing that iteration we increase number of 'handled' rows/cols by 1:
//
//   0 0 0 0 0 0 0 0
//   0 0 0 0 0 0 0 0
//   0 0 0 0 0 0 0 0
//   0 0 0 v v v v v
//   0 0 0 v v v v v
//   0 0 0 v v v v v
//   0 0 0 v v v v v
//   0 0 0 v v v v v
//
// NOTE: This implementation does decomposition in-place, since 'L' is guaranteed to have
// ones on the diagonal we can store all meaningfull elements of 'L' and 'U' inside one matrix
//
template <typename T>
void LU_serial(T *A, const size_t ROWS, const size_t COLS) {
	for (size_t i = 0; i < std::min(ROWS - 1, COLS); ++i) {
		// 1)
		const T inverseAii = T(1) / A[i * COLS + i];

		for (size_t j = i + 1; j < ROWS; ++j)
			A[j * COLS + i] *= inverseAii;

		// 2)
		for (size_t j = i + 1; j < ROWS; ++j)
			for (size_t k = i + 1; k < COLS; ++k)
				A[j * COLS + k] -= A[j * COLS + i] * A[i * COLS + k];
	}
}


// # Block LU (serial) #
// - No pivoting
// - Time complexity O(const * N^3)
//
// Similar in concept to regular LU, the difference is that instead of moving forward by
// 1 row/column per iteration, we move forward by BLOCK_SIZE
//
// Being essentially an LU rewriten using matrix operations, blocked version improves cache
// friendliness improving serial performance by ~2-3 times (depending on the compiler and hardware)
//
// NOTE: To benefit from improved cache use, blocks need to be copied to separate storages before
//       doing any operations on them. The same method can be done in-place similarly to regular LU,
//       however that defeats the purpose of blocking
//
// In subsequent descriptions, following notation is used to denote matrix blocks:
//
//   |-----|-----|-----|
//   | A11 | A12 | A13 |   
//   |-----|-----|-----|   <
//   | A21 | A22 | A23 |   < BLOCK_SIZE
//   |-----|-----|-----|   <
//   | A31 | A32 | A33 |
//   |-----|-----|-----|
//
//          ^ ^ ^ BLOCK_SIZE
//
// Here, blocks(A11), (A12), (A13), (A21), (A31)containg already handled elements,
// while iterations perform matrix operations on blocks (A22), (A23), (A32), (A33)
//
// >>> Method goes over rows with following iterations:
//
// 1) Perform regular LU decomposition on a vertical block (A23 & A32)
//
//   0 0 0 0 0 0 0 0   (zeroes represent already 'handled' elements) 
//   0 0 0 0 0 0 0 0
//   0 0 # # v v v v   ('#' represents elements that are changed on a current step)
//   0 0 # # v v v v
//   0 0 # # v v v v
//   0 0 # # v v v v
//   0 0 # # v v v v
//   0 0 # # v v v v

// 2) Go over columns in block (A23) solving SLAEs '(U22)*x = <column of A23>'
//    and placing solutions in place of said columns
//
// Here, (U22) denotes upper-triangular part of (A22)
//
//   0 0 0 0 0 0 0 0
//   0 0 0 0 0 0 0 0
//   0 0 M M # # # #   <- elements 'M' here are already handled, but are marked differently
//   0 0 0 M # # # #      to represent block (U22) which we use as SLAE matrix
//   0 0 0 0 v v v v
//   0 0 0 0 v v v v
//   0 0 0 0 v v v v
//   0 0 0 0 v v v v
//
// NOTE: SLAE solution requires only backwards Gauss elimination due to the structure of (U22)
//
// 3) Multiply blocks (A32) and (A23) and substract the result from (A33),
//    aka 'A33 -= A32 * A23'
//
//   0 0 0 0 0 0 0 0
//   0 0 0 0 0 0 0 0
//   0 0 0 0 U U U U   <- elements 'L' and 'U' here are already handled, but are marked differently
//   0 0 0 0 U U U U      to represent blocks (A32) and (A23) which we multiply by each other
//   0 0 L L # # # #
//   0 0 L L # # # #
//   0 0 L L # # # #
//   0 0 L L # # # #
//
// NOTE: This step takes the vast majority of time (~90%) to complete
//
// NOTE: Since we are copying blocks (A32) and (A23) anyway, it makes sense to transpose one of them
//       during copy and use a more cache friendly "row-major by col-major" matrix multiplication,
//       improving overall performance
//
template <typename T>
void blockLU_serial(T *A, const size_t N, const size_t b) {
	const size_t total_length = N * b + b * (N - b);

	T* const buffer = new T[total_length * sizeof(T)];

	T *A_22;
	T *A_32;
	T *A_23; // NOTE: A_23 is col-major!

	for (size_t i = 0; i < N; i += b) {
		// Adjust pointers. Blocks A_22 -> A_32 -> A_23 are stored contiguously in corresponding order
		const size_t rows_22 = b;
		const size_t cols_22 = b;

		const size_t rows_32 = N - b - i;
		const size_t cols_32 = b;

		const size_t rows_23 = b;
		const size_t cols_23 = N - b - i;

		A_22 = buffer;
		A_32 = A_22 + rows_22 * cols_22;
		A_23 = A_32 + rows_32 * cols_32;

		// 1)
		// Find LU decomposition of block (A22 & A32)
		span_copy_rm_to_rm(
			// source
			A, N, N,
			i, i,
			rows_22 + rows_32, cols_22,
			// dest
			A_22, rows_22 + rows_32, cols_22,
			0, 0
		);

		LU_serial(A_22, rows_22 + rows_32, cols_22);

		span_copy_rm_to_rm(
			// source
			A_22, rows_22 + rows_32, cols_22,
			0, 0,
			rows_22 + rows_32, cols_22,
			// dest
			A, N, N,
			i, i
		);

		// 2)
		// Solve (N - b - i) systems U22*x = <column of A23>
		// to get A23 = L22^-1 * A23
		span_copy_rm_to_cm(
			// source
			A, N, N,
			i, i + b,
			b, N - b - i,
			// dest
			A_23, rows_23, cols_23,
			0, 0
		);

		span_get_U23(
			A_22, rows_22, cols_22,
			A_23, rows_23, cols_23
		);

		span_copy_cm_to_rm(
			// source
			A_23, rows_23, cols_23,
			0, 0,
			rows_23, cols_23,
			// dest
			A, N, N,
			i, i + b
		);

		// 3)
		// A33 -= A32 * A23
		span_substract_product(
			// source 1
			A_32, rows_32, cols_32,
			// source 2
			A_23, rows_23, cols_23,
			// dest
			A, N, N,
			i + b, i + b
		);
	}

	delete[] buffer;
}