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

#include <complex>
#include <vector>
#include <cmath>
#include <omp.h>

#include "fft.hpp"

const double PI = 3.141592653589793238460;

#ifdef ZFP_RECURSIVE_FORM
template <typename T>
void fft(std::vector<std::complex<T>>& a) {
    int n = a.size();
    if (n <= 1) return;

    std::vector<std::complex<T>> even(n / 2);
    std::vector<std::complex<T>> odd(n / 2);

    #pragma omp parallel for
    for (int i = 0; i < n / 2; ++i) {
        even[i] = a[i * 2];
        odd[i] = a[i * 2 + 1];
    }

    fft(even);
    fft(odd);

    #pragma omp parallel for
    for (int k = 0; k < n / 2; ++k) {
        std::complex<T> t = std::polar(T(1.0), T(-2 * PI * k / n)) * odd[k];
        a[k] = even[k] + t;
        a[k + n / 2] = even[k] - t;
    }
}
#else
template <typename T>
void fft(std::vector<std::complex<T>>& a) {
    int n = a.size();
    if (n <= 1) return;

    // Bit-reversal permutation
    int log2n = std::log2(n);
    for (int i = 0; i < n; ++i) {
        int rev = 0;
        for (int j = 0; j < log2n; ++j) {
            if (i & (1 << j)) {
                rev |= 1 << (log2n - 1 - j);
            }
        }
        if (i < rev) {
            std::swap(a[i], a[rev]);
        }
    }

    // Cooley-Tukey FFT
    for (int m = 2; m <= n; m *= 2) {
        std::complex<T> wm = std::polar(T(1.0), T(-2 * PI / m));
        #pragma omp parallel for
        for (int k = 0; k < n; k += m) {
            std::complex<T> w = 1;
            for (int j = 0; j < m / 2; ++j) {
                std::complex<T> t = w * a[k + j + m / 2];
                std::complex<T> u = a[k + j];
                a[k + j] = u + t;
                a[k + j + m / 2] = u - t;
                w *= wm;
            }
        }
    }
}
#endif

// Explicit template instantiation
template void fft<float>(std::vector<std::complex<float>>&);
template void fft<double>(std::vector<std::complex<double>>&);
