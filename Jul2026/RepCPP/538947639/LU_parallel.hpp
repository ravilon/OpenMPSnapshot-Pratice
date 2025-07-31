#pragma once

#include "matrix_span_parallel.hpp"
#include "LU_serial.hpp"


// # LU (parallel) #
// >>> For full method description and notes check serial version in "LU_serial.hpp"
//
// NOTE: The only meaningful parallelization that can be done in regular LU is step in 2),
//       since iterations over rows on that step are independent and can be done in any order,
//       we can parallelize that loop with OpenMP
//
template <typename T>
void LU_parallel(T *A, const int ROWS, const int COLS) {
	for (int i = 0; i < std::min(ROWS - 1, COLS); ++i) {
		// 1)
		const T inverseAii = T(1) / A[i * COLS + i];

		for (int j = i + 1; j < ROWS; ++j)
			A[j * COLS + i] *= inverseAii;

		// 2)
		#pragma omp parallel for schedule(static) firstprivate(A, i, ROWS, COLS)
		for (int j = i + 1; j < ROWS; ++j)
			for (int k = i + 1; k < COLS; ++k)
				A[j * COLS + k] -= A[j * COLS + i] * A[i * COLS + k];
	}
}


// ### Block LU (parallel) ###
// >>> For full method description and notes check serial version in "LU_serial.hpp"
//
// NOTE: Out main culprit for parallelization is matrix multiplication in step 3),
//       with OpenMP we can simply parallelize the outer-most loop 
//
// NOTE: While less significant, step 2) can also be parallelized since all SLAE
//       solutions on that step are independent and can be done in any order
//
template <typename T>
void blockLU_parallel(T *A, const size_t N, const size_t b) {
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
		parspan_copy_rm_to_rm(
			// source
			A, N, N,
			i, i,
			rows_22 + rows_32, cols_22,
			// dest
			A_22, rows_22 + rows_32, cols_22,
			0, 0
		);

		LU_serial(A_22, rows_22 + rows_32, cols_22);
			// for any sensible block size A_22 & A_32 is
			// too small to warrant parallelization
		
		parspan_copy_rm_to_rm(
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
		parspan_copy_rm_to_cm(
			// source
			A, N, N,
			i, i + b,
			b, N - b - i,
			// dest
			A_23, rows_23, cols_23,
			0, 0
		);

		parspan_get_U23(
			A_22, rows_22, cols_22,
			A_23, rows_23, cols_23
		);

		parspan_copy_cm_to_rm(
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
		parspan_substract_product(
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