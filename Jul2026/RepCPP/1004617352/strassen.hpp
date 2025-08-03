#pragma once

#include <vector>

using Matrix = std::vector<std::vector<long double>>;

Matrix add(const Matrix& A, const Matrix& B);

Matrix subtract(const Matrix& A, const Matrix& B);

Matrix strassen(const Matrix& A, const Matrix& B);