#pragma once

#include <omp.h>


template<typename T>
inline void parspan_copy_rm_to_rm(
const T *src, size_t src_rows, size_t src_cols,
size_t src_span_i, size_t src_span_j, size_t src_span_rows, size_t src_span_cols,
T *dst, size_t dst_rows, size_t dst_cols,
size_t dst_span_i, size_t dst_span_j) {

for (size_t i = 0; i < src_span_rows; ++i)
for (size_t j = 0; j < src_span_cols; ++j)
dst[(dst_span_i + i) * dst_cols + dst_span_j + j] = src[(src_span_i + i) * src_cols + src_span_j + j];
}


template<typename T>
inline void parspan_copy_rm_to_cm(
const T *src, size_t src_rows, size_t src_cols,
size_t src_span_i, size_t src_span_j, size_t src_span_rows, size_t src_span_cols,
T *dst, size_t dst_rows, size_t dst_cols,
size_t dst_span_i, size_t dst_span_j) {

for (size_t i = 0; i < src_span_rows; ++i)
for (size_t j = 0; j < src_span_cols; ++j)
dst[(dst_span_j + j) * dst_rows + dst_span_i + i] = src[(src_span_i + i) * src_cols + src_span_j + j];
}


template<typename T>
inline void parspan_copy_cm_to_rm(
const T *src, size_t src_rows, size_t src_cols,
size_t src_span_i, size_t src_span_j, size_t src_span_rows, size_t src_span_cols,
T *dst, size_t dst_rows, size_t dst_cols,
size_t dst_span_i, size_t dst_span_j) {

for (size_t i = 0; i < src_span_rows; ++i)
for (size_t j = 0; j < src_span_cols; ++j)
dst[(dst_span_i + i) * dst_cols + dst_span_j + j] = src[(src_span_j + j) * src_rows + src_span_j + i];
}


template<typename T>
inline void parspan_get_U23(
const T *src1, size_t src1_rows, size_t src1_cols,
T *src2, size_t src2_rows, size_t src2_cols) {

// Not using 'span' here since we operate with blocks that start at (0, 0)
// NOTE: src2 is assumed to be col-major
#pragma omp parallel for schedule(static) firstprivate(src1, src1_rows, src1_cols, src2, src2_rows, src2_cols)
for (int j = 0; j < src2_cols; ++j)
for (int i = 0; i < src2_rows; ++i)
for (int k = i - 1; k >= 0; --k)
src2[j * src2_rows + i] -= src2[j * src2_rows + k] * src1[i * src1_cols + k];
// Effect of parallelization isn't very noticable here but seems to be positive for large sizes
}


template <typename T>
inline void parspan_substract_product(
T const *src1, int src1_rows, int src1_cols,
T const *src2, int src2_rows, int src2_cols,
T *dst, int dst_rows, int dst_cols,
int dst_i, int dst_j) {

// NOTE: src2 is assumed to be col-major
// (aka transposed matrix used in multiplication)
T temp;

int shift_src1;
int shift_src2;
int shift_dst;

#pragma omp parallel for schedule(static) firstprivate(src1, src2, dst, temp, shift_src1, shift_src2, shift_dst, src1_rows, src1_cols, src2_rows, src2_cols, dst_i, dst_j)
for (int i = 0; i < src1_rows; ++i) {
shift_dst = (dst_i + i) * dst_cols + dst_j;
shift_src1 = i * src1_cols;

for (int j = 0; j < src2_cols; ++j) {
shift_src2 = j * src2_rows;

temp = T(0);
for (int k = 0; k < src1_cols; ++k) temp += src1[shift_src1 + k] * src2[shift_src2 + k];
dst[shift_dst + j] -= temp;
}
}
}