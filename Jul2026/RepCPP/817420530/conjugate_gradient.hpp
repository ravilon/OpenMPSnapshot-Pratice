#ifndef ZFP_CONJUGATE_GRADIENT_HPP
#define ZFP_CONJUGATE_GRADIENT_HPP
#pragma once

#include <vector>

// This function solves a system of linear equations Ax = b using the
// Conjugate Gradient method. The function takes the matrix A, the vector b,
// and an initial guess for the solution x. The function returns the solution
// x after a specified number of iterations or when the solution converges
// to within a specified tolerance.
template <typename T>
void conjugateGradient(const std::vector<std::vector<T>>& A, 
                       const std::vector<T>& b, std::vector<T>& x, 
                       int max_iter = 1000, T tol = 1e-10);

#endif // ZFP_CONJUGATE_GRADIENT_HPP
