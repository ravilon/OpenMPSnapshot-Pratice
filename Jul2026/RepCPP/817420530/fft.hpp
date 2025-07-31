
#ifndef ZFP_FFT_HPP
#define ZFP_FFT_HPP
#pragma once

#include <complex>
#include <vector>

// This implementation of the Fast Fourier Transform (FFT) algorithm uses OpenMP
// to parallelize the computation of the FFT. The algorithm can be implemented
// in a recursive or iterative form.
// Define ZFP_RECURSIVE_FORM to use the recursive form.
template <typename T>
void fft(std::vector<std::complex<T>>& x);

#endif // ZFP_FFT_HPP
