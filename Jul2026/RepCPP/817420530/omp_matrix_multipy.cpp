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
#include <omp.h>

#include "matrix_multipy.hpp"

template <typename T>
void matrixMultiply(const std::vector<std::vector<T>>& A, 
                    const std::vector<std::vector<T>>& B,
                    std::vector<std::vector<T>>& C) {

    int n = A.size();
    if (n == 0) return;
    int m = A[0].size();
    int p = B[0].size();

    // Ensure B has the correct dimensions
    if (m != B.size()) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    // Resize C to be n x p and initialize with zeros
    C.resize(n, std::vector<T>(p, 0));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            T sum = 0;
            for (int k = 0; k < m; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

// Explicit template instantiation
template void matrixMultiply<int>(const std::vector<std::vector<int>>&, 
                                  const std::vector<std::vector<int>>&, 
                                  std::vector<std::vector<int>>&);
template void matrixMultiply<float>(const std::vector<std::vector<float>>&, 
                                    const std::vector<std::vector<float>>&, 
                                    std::vector<std::vector<float>>&);
template void matrixMultiply<double>(const std::vector<std::vector<double>>&, 
                                     const std::vector<std::vector<double>>&, 
                                     std::vector<std::vector<double>>&);
template void matrixMultiply<long>(const std::vector<std::vector<long>>&,
                                    const std::vector<std::vector<long>>&,
                                    std::vector<std::vector<long>>&);
