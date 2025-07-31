#pragma once

#include "blas.hpp"
#include "csmat.hpp"

#include <utility>
#include <vector>

namespace algorithms {
    /// This function implements the Stochastic Approach for Link-Structure Analysis (SALSA) algorithm.
    /// It computes two scoring vectors: authority and hub. These represent the authoriy and hub scores
    /// for each node of the sparse matrix used as input.
    ///
    /// @param mtx The input sparse matrix.
    /// @param error The epsilon threshold used to determine when the algorithm has converged.
    /// @return A tuple containing the authority and hub scoring vectors.
    template <typename T>
    auto salsa(CsMat<T> const& A) -> std::pair<std::vector<T>, std::vector<T>> {
        // A_r <- for each A row r: r / |r|
        CsMat<T> A_r = blas::normalize_rows(A);

        // A_cT <- A_c'
        CsMat<T> AT = A.transpose();

        // A_c <- for each A col c: c / |c|
        CsMat<T> A_cT = blas::normalize_rows(AT);

        // A_tilde <- A_c' * A_r.
        CsMat<T> A_tilde = blas::spmm(A_cT, A_r);

        // H_tilde <- A_r * A_c'.
        CsMat<T> H_tilde = blas::spmm(A_r, A_cT);

        // TODO: Compute the dominant eigenvectors for each tilde matrix.
        // std::vector<T> a = blas::dominant_eigenvector(A_tilde);
        // std::vector<T> h = blas::dominant_eigenvector(H_tilde);

        // return std::make_pair(a, h);
    }
} // namespace algorithms