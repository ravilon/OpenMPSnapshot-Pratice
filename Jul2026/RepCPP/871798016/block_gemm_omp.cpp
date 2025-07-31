#include <vector>
#include <immintrin.h>

#include "block_gemm_omp.h"

void transpose_matrix(std::vector<float>& a, int n) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      std::swap(a[i * n + j], a[j * n + i]);
    }
  }
}

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b, int n) {
  std::vector<float> b_T(b);
  transpose_matrix(b_T, n);

  constexpr int block_size = 32;
  std::vector<float> result(n * n, 0.0f);

#pragma omp parallel for collapse(2)
  for (int block_i = 0; block_i < n; block_i += block_size) {
    for (int block_j = 0; block_j < n; block_j += block_size) {
      for (int block_k = 0; block_k < n; block_k += block_size) {

        float a_block[block_size][block_size];
        float b_block[block_size][block_size];

        for (int i = 0; i < block_size; ++i) {
          for (int k = 0; k < block_size; ++k) {
            a_block[i][k] = a[(block_i + i) * n + block_k + k];
            b_block[i][k] = b_T[(block_j + i) * n + block_k + k];
          }
        }

        for (int i = 0; i < block_size; ++i) {
          for (int j = 0; j < block_size; ++j) {
            for (int k = 0; k < block_size; ++k) {
              result[(block_i + i) * n + block_j + j] +=
                  a_block[i][k] * b_block[j][k];
            }
          }
        }
      }
    }
  }

  return result;
}
