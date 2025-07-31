#pragma once
#include <omp.h>
#include <vector>

#include "./monolish_solver.hpp"
#include "monolish/common/monolish_common.hpp"

namespace monolish {
/**
 * @brief handling eigenvalues and eigenvectors
 **/
namespace standard_eigen {

/**
 * @addtogroup sEigen
 * @{
 */

/**
 * \defgroup sLOBPCG monolish::standard_eigen::LOBPCG
 * @brief LOBPCG solver
 * @{
 */
/**
 * @brief LOBPCG solver
 * @note
 * attribute:
 * - solver : true
 * - preconditioner : false
 * @note
 * input / architecture
 * - Dense / Intel : true
 * - Dense / NVIDIA : true
 * - Dense / OSS : true
 * - Sparse / Intel : true
 * - Sparse / NVIDIA : true
 * - Sparse / OSS : true
 */
template <typename MATRIX, typename Float>
class LOBPCG : public solver::solver<MATRIX, Float> {
private:
  // TODO: support multiple lambda(eigenvalue)s
  [[nodiscard]] int monolish_LOBPCG(MATRIX &A, vector<Float> &lambda,
                                    matrix::Dense<Float> &x);

public:
  /**
   * @brief calculate eigenvalues and eigenvectors or A by LOBPCG method(lib=0:
   *monolish)
   * @param[in] A CRS format Matrix
   * @param[in] lambda up to m smallest eigenvalue
   * @param[in] x corresponding eigenvectors in Dense matrix format
   * @return error code (only 0 now)
   **/
  [[nodiscard]] int solve(MATRIX &A, vector<Float> &lambda,
                          matrix::Dense<Float> &x);

  void create_precond(MATRIX &A) {
    throw std::runtime_error("this precond. is not impl.");
  }

  void apply_precond(const vector<Float> &r, vector<Float> &z) {
    throw std::runtime_error("this precond. is not impl.");
  }

  /**
   * @brief get solver name "monolish::standard_eigen::LOBPCG"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::string name() const {
    return "monolish::standard_eigen::LOBPCG";
  }

  /**
   * @brief get solver name "LOBPCG"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::string solver_name() const { return "LOBPCG"; }
};
/**@}*/

/**
 * \defgroup sDC monolish::standard_eigen::DC
 * @brief Devide and Conquer solver
 * @{
 */
/**
 * @brief Devide and Conquer solver
 * @note
 * attribute:
 * - solver : true
 * - preconditioner : false
 * @note
 * input / architecture
 * - Dense / Intel : true
 * - Dense / NVIDIA : true
 * - Dense / OSS : true
 * - Sparse / Intel : false
 * - Sparse / NVIDIA : false
 * - Sparse / OSS : false
 */
template <typename MATRIX, typename Float>
class DC : public solver::solver<MATRIX, Float> {
private:
  [[nodiscard]] int LAPACK_DC(MATRIX &A, vector<Float> &lambda);

public:
  [[nodiscard]] int solve(MATRIX &A, vector<Float> &lambda);

  void create_precond(MATRIX &A) {
    throw std::runtime_error("this precond. is not impl.");
  }

  void apply_precond(const vector<Float> &r, vector<Float> &z) {
    throw std::runtime_error("this precond. is not impl.");
  }

  /**
   * @brief get solver name "monolish::standard_eigen::DC"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::string name() const {
    return "monolish::standard_eigen::DC";
  }

  /**
   * @brief get solver name "DC"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::string solver_name() const { return "DC"; }
};
/**@}*/
} // namespace standard_eigen
} // namespace monolish
