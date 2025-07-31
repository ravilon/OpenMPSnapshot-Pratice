// Copyright 2023 Paolo Fabio Zaino
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>
#include <cmath>
#include <omp.h>
#include <stdexcept>

#include "conjugate_gradient.hpp"

template <typename T>
void conjugateGradient(const std::vector<std::vector<T>>& A, 
                       const std::vector<T>& b, std::vector<T>& x, 
                       int max_iter, T tol) 
{
    int n = b.size();
    
    if (A.size() != n || A[0].size() != n) {
        throw std::invalid_argument("Matrix A must be square and the size of A must match the size of vector b.");
    }
    
    std::vector<T> r(n), p(n), Ap(n);
    T alpha, beta, rsold, rsnew;

    // r = b - A * x
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        r[i] = b[i];
        for (int j = 0; j < n; ++j) {
            r[i] -= A[i][j] * x[j];
        }
    }
    p = r;
    rsold = 0;
    #pragma omp parallel for reduction(+:rsold)
    for (int i = 0; i < n; ++i) {
        rsold += r[i] * r[i];
    }

    for (int iter = 0; iter < max_iter; ++iter) {
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            Ap[i] = 0;
            for (int j = 0; j < n; ++j) {
                Ap[i] += A[i][j] * p[j];
            }
        }

        T pAp = 0;
        #pragma omp parallel for reduction(+:pAp)
        for (int i = 0; i < n; ++i) {
            pAp += p[i] * Ap[i];
        }

        if (pAp == 0) {
            throw std::runtime_error("Division by zero in alpha calculation. The input matrix might be singular.");
        }

        alpha = rsold / pAp;

        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        rsnew = 0;
        #pragma omp parallel for reduction(+:rsnew)
        for (int i = 0; i < n; ++i) {
            rsnew += r[i] * r[i];
        }

        if (std::sqrt(rsnew) < tol) break;

        beta = rsnew / rsold;

        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            p[i] = r[i] + beta * p[i];
        }

        rsold = rsnew;
    }
}

// Explicit template instantiation
template void conjugateGradient<float>(const std::vector<std::vector<float>>&, const std::vector<float>&, std::vector<float>&, int, float);
template void conjugateGradient<double>(const std::vector<std::vector<double>>&, const std::vector<double>&, std::vector<double>&, int, double);
template void conjugateGradient<long>(const std::vector<std::vector<long>>&, const std::vector<long>&, std::vector<long>&, int, long);
