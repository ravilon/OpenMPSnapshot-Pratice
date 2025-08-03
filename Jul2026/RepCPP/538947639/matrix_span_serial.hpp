#pragma once


template<typename T>
inline void span_copy_rm_to_rm(
const T *src, size_t src_rows, size_t src_cols,
size_t src_span_i, size_t src_span_j, size_t src_span_rows, size_t src_span_cols,
T *dst, size_t dst_rows, size_t dst_cols,
size_t dst_span_i, size_t dst_span_j) {

for (size_t i = 0; i < src_span_rows; ++i)
for (size_t j = 0; j < src_span_cols; ++j)
dst[(dst_span_i + i) * dst_cols + dst_span_j + j] = src[(src_span_i + i) * src_cols + src_span_j + j];
}


template<typename T>
inline void span_copy_rm_to_cm(
const T *src, size_t src_rows, size_t src_cols,
size_t src_span_i, size_t src_span_j, size_t src_span_rows, size_t src_span_cols,
T *dst, size_t dst_rows, size_t dst_cols,
size_t dst_span_i, size_t dst_span_j) {

for (size_t i = 0; i < src_span_rows; ++i)
for (size_t j = 0; j < src_span_cols; ++j)
dst[(dst_span_j + j) * dst_rows + dst_span_i + i] = src[(src_span_i + i) * src_cols + src_span_j + j];
}


template<typename T>
inline void span_copy_cm_to_rm(
const T *src, size_t src_rows, size_t src_cols,
size_t src_span_i, size_t src_span_j, size_t src_span_rows, size_t src_span_cols,
T *dst, size_t dst_rows, size_t dst_cols,
size_t dst_span_i, size_t dst_span_j) {

for (size_t i = 0; i < src_span_rows; ++i)
for (size_t j = 0; j < src_span_cols; ++j)
dst[(dst_span_i + i) * dst_cols + dst_span_j + j] = src[(src_span_j + j) * src_rows + src_span_j + i];
}


template<typename T>
inline void span_get_U23(
const T *src1, size_t src1_rows, size_t src1_cols,
T *src2, size_t src2_rows, size_t src2_cols) {

// Not using 'span' here since we operate with blocks that start at (0, 0)
// NOTE: src2 is assumed to be col-major
for (size_t j = 0; j < src2_cols; ++j)
for (size_t i = 0; i < src2_rows; ++i)
for (int k = i - 1; k >= 0; --k)
src2[j * src2_rows + i] -= src2[j * src2_rows + k] * src1[i * src1_cols + k];
}


template <typename T>
inline void span_substract_product(
T const *src1, size_t src1_rows, size_t src1_cols,
T const *src2, size_t src2_rows, size_t src2_cols,
T *dst, size_t dst_rows, size_t dst_cols,
size_t dst_i, size_t dst_j) {

// NOTE: src2 is assumed to be col-major
// (aka transposed matrix used in multiplication)
T temp;

size_t shift_src1;
size_t shift_src2;
size_t shift_dst;

for (size_t i = 0; i < src1_rows; ++i) {
shift_dst = (dst_i + i) * dst_cols + dst_j;
shift_src1 = i * src1_cols;

for (size_t j = 0; j < src2_cols; ++j) {
shift_src2 = j * src2_rows;

temp = T(0);
for (size_t k = 0; k < src1_cols; ++k) temp += src1[shift_src1 + k] * src2[shift_src2 + k];
dst[shift_dst + j] -= temp;
}
}
}