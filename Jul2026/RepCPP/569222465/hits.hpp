#pragma once

#include "blas.hpp"
#include "csmat.hpp"

#include <utility>
#include <vector>

namespace algorithms {
    /// This function implements the Hint-Induced Topic Search (HITS) algorithm.
    /// It computes two scoring vectors: authority and hub. These represent the authoriy and hub scores
    /// for each node of the sparse matrix used as input.
    ///
    /// @param mtx The input sparse matrix.
    /// @param error The epsilon threshold used to determine when the algorithm has converged.
    /// @return A tuple containing the authority and hub scoring vectors.
    template <typename T = double>
    auto hits(CsMat<T> const& A, double error) -> std::pair<std::vector<T>, std::vector<T>> {
        size_t const n = A.get_nrows();

        // Authority scoring
        std::vector<T> a(n, 1.0);
        // Hub scoring
        std::vector<T> h(n, 1.0);

        // Transpose the input matrix once as it is constant
        CsMat<T> AT = A.transposed();

        // Forward declare the norms of the authority and hub scoring vectors at the previous iteration
        T norm_a_prev;
        T norm_h_prev;

        // Repeat until convergence
        do {
            // Store previous iteration by calling the copy constructor
            std::vector<T> a_prev = a;
            std::vector<T> h_prev = h;

            // a <- Ah
            blas::spmv(1.0, A, h, 0.0, a);
            // a <- a / |a|
            blas::normalize(a);

            // h <- A'a
            blas::spmv(1.0, AT, a, 0.0, h);
            // h <- h / |h|
            blas::normalize(h);

            norm_a_prev = blas::norm2(a_prev);
            norm_h_prev = blas::norm2(h_prev);
        } while (blas::norm2(a) - norm_a_prev > error && blas::norm2(h) - norm_h_prev > error);

        return std::make_pair(a, h);
    }
} // namespace algorithms
