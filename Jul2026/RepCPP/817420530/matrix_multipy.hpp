#ifndef ZFP_MATRIX_MULTIPLY_HPP
#define ZFP_MATRIX_MULTIPLY_HPP
#pragma once

#include <vector>

template <typename T>
void matrixMultiply(const std::vector<std::vector<T>>& A, 
                    const std::vector<std::vector<T>>& B,
                    std::vector<std::vector<T>>& C);

#endif // ZFP_MATRIX_MULTIPLY_HPP
