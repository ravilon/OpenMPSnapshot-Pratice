#pragma once

#include "csmat.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <omp.h>

namespace blas {
    /**
     * Computes the norm-2 of a vector.
     *
     * @param x An input vector.
     * @return The norm-2 of `x`.
     **/
    template <typename T>
    auto norm2(std::vector<T> const& x) -> T {
        return sqrt(std::reduce(x.begin(), x.end(), 0.0));
    }

    /**
     * Normalizes a vector.
     *
     * @param x An input vector to normalize.
     **/
    template <typename T>
    auto normalize(std::vector<T>& x) -> void {
        T norm = norm2(x);
        #if defined(PARALLEL)
        #pragma omp parallel
        #endif
        std::for_each(x.begin(), x.end(), [norm](T& v) {
            v /= norm;
        });
    }

    /**
     * Normalizes the rows of a CSR matrix.
     *
     * @param A An input CSR matrix.
     * @return The input CSR matrix with its rows normalized.
     **/
    template <typename T>
    auto normalize_rows(CsMat<T> const& A) -> CsMat<T> {
        CsMat<T> A_normalized_rows = A;

        // Loop over each row in the matrix
        #if defined(PARALLEL)
        #pragma omp parallel for
        #endif
        for (size_t i = 0; i < A.get_nrows(); ++i) {
            std::vector<T> row_data;
            // Extract the non-zero values in the current row
            #pragma omp simd
            for (size_t j = A.get_indptr()[i]; j < A.get_indptr()[i + 1]; ++j) {
                row_data.push_back(A.get_data()[j]);
            }

            // Normalize the current row
            normalize(row_data);

            // Update the matrix data with the normalized values
            #pragma omp simd
            for (size_t j = A.get_indptr()[i]; j < A.get_indptr()[i + 1]; ++j) {
                A_normalized_rows.get_mut_data()[j] = row_data[j - A.get_indptr()[i]];
            }
        }

        return A_normalized_rows;
    }

    /**
     * Sparse matrix-vector multiplication for the Compressed Sparse Row storage format.
     *
     * @param alpha A constant.
     * @param A A matrix in the CSR format.
     * @param x A vector.
     * @param beta A constant.
     * @param y The output vector.
     **/
    template <typename T>
    auto spmv(
        T alpha,
        CsMat<T> const& A,
        std::vector<T> const& x,
        T beta,
        std::vector<T>& y
    ) -> void {
        assert(A.get_ncols() == x.size());
        assert(A.get_ncols() == y.size());

        std::vector<T> const& values = A.get_data();
        std::vector<size_t> const& indices = A.get_indices();
        std::vector<size_t> const& indptr = A.get_indptr();

        #if defined(PARALLEL)
        #pragma omp parallel for
        #endif
        for (size_t i = 0; i < A.get_nrows(); ++i) {
            T tmp = y[i] * beta;
            #pragma omp simd
            for (size_t j = indptr[i]; j < indptr[i + 1]; ++j) {
                tmp += alpha * values[j] * x[indices[j]];
            }
            y[i] += tmp;
        }
    }

    template <typename T>
    auto spmm(CsMat<T> const& A, CsMat<T> const& B) -> CsMat<T> {
        // Check that the matrices have compatible dimensions
        assert(A.get_ncols() == B.get_nrows());

        CsMat<T> C;
        C.get_mut_nrows() = A.get_nrows();
        C.get_mut_ncols() = B.get_ncols();

        // Compute the row pointer array for C
        C.get_mut_indptr().push_back(0);
        #if defined(PARALLEL)
        #pragma omp parallel for
        #endif
        for (size_t i = 0; i < A.get_nrows(); i++) {
            std::vector<T> crow(B.get_ncols(), 0.0);
            for (size_t j = A.get_indptr()[i]; j < A.get_indptr()[i + 1]; j++) {
                T val = A.get_data()[j];
                #pragma omp simd
                for (
                    size_t k = B.get_indptr()[A.get_indices()[j]];
                    k < B.get_indptr()[A.get_indices()[j] + 1];
                    k++
                ) {
                    crow[B.get_indices()[k]] += val * B.get_data()[k];
                }
            }

            #pragma omp simd
            for (size_t j = 0; j < B.get_ncols(); j++) {
                if (crow[j] != 0.0) {
                    C.get_mut_data().push_back(crow[j]);
                    C.get_mut_indices().push_back(j);
                }
            }
            C.get_mut_indptr().push_back(C.get_indices().size());
        }

        return C;
    }
} // namespace blas
