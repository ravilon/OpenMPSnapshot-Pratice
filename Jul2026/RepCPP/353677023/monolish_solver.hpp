#pragma once
#include <vector>

#include "monolish/common/monolish_common.hpp"
#include <functional>

namespace monolish {
/**
 * @brief Linear solver base class
 **/
namespace solver {

/**
 * @brief Enum class defining how to handle initial vectors
 * RANDOM use randomly initialized vectors
 * USER use initial vectors set by user
 */
enum class initvec_scheme {
  RANDOM,
  USER,
};

template <typename MATRIX, typename Float> class precondition;

/**
 * @addtogroup solver_base
 * @{
 */
/**
 * @brief solver base class
 **/
template <typename MATRIX, typename Float> class solver {
private:
protected:
  int lib = 0;
  double tol = 1.0e-8;
  size_t miniter = 0;
  size_t maxiter = SIZE_MAX;
  size_t resid_method = 0;
  bool print_rhistory = false;
  std::string rhistory_file;
  std::ostream *rhistory_stream;
  initvec_scheme initvecscheme = initvec_scheme::RANDOM;

  double final_resid = 0;
  size_t final_iter = 0;

  Float omega = 1.9; // for SOR
  int singularity;   // for sparse LU/QR/Cholesky
  int reorder = 3;   // for sparse LU/QR/Cholesky;

  Float get_residual(vector<Float> &x);
  precondition<MATRIX, Float> precond;

public:
  /**
   * @brief create solver class
   **/
  solver(){};

  /**
   * @brief delete solver class
   **/
  ~solver() {
    if (rhistory_stream != &std::cout && rhistory_file.empty() != true) {
      delete rhistory_stream;
    }
  }

  /**
   * @brief set precondition create function
   * @param[in] p solver class for precondition
   **/
  template <class PRECOND> void set_create_precond(PRECOND &p);

  /**
   * @brief set precondition apply function
   * @param[in] p solver class for precondition
   **/
  template <class PRECOND> void set_apply_precond(PRECOND &p);

  /**
   * @brief set library option (to change library, monolish, cusolver, etc.)
   * @param[in] l library number
   **/
  void set_lib(int l) { lib = l; }

  /**
   * @brief set tolerance (default:1.0e-8)
   * @param[in] t tolerance
   **/
  void set_tol(double t) { tol = t; }

  /**
   * @brief set max iter. (default = SIZE_MAX)
   * @param[in] max maxiter
   **/
  void set_maxiter(size_t max) { maxiter = max; }

  /**
   * @brief set min iter. (default = 0)
   * @param[in] min miniter
   **/
  void set_miniter(size_t min) { miniter = min; }

  /**
   * @brief set residual method (default=0)
   * @param[in] r residual method number (0:nrm2)
   **/
  void set_residual_method(size_t r) { resid_method = r; }

  /**
   * @brief print rhistory to standart out true/false. (default = false)
   * @param[in] flag
   **/
  void set_print_rhistory(bool flag) {
    print_rhistory = flag;
    rhistory_stream = &std::cout;
  }

  /**
   * @brief rhistory filename
   * @param[in] file: output file name
   **/
  void set_rhistory_filename(std::string file) {
    rhistory_file = file;

    // file open
    rhistory_stream = new std::ofstream(rhistory_file);
    if (rhistory_stream->fail()) {
      throw std::runtime_error("error bad filename");
    }
  }

  /**
   * @brief set how to handle initial vector
   * @param[in] scheme: RANDOM or USER
   */
  void set_initvec_scheme(initvec_scheme scheme) { initvecscheme = scheme; }
  ///////////////////////////////////////////////////////////////////

  /**
   * @brief get library option
   * @return library number
   **/
  [[nodiscard]] int get_lib() const { return lib; }

  /**
   * @brief get tolerance
   * @return tolerance
   **/
  [[nodiscard]] double get_tol() const { return tol; }

  /**
   * @brief get maxiter
   * @return  maxiter
   **/
  [[nodiscard]] size_t get_maxiter() const { return maxiter; }

  /**
   * @brief get miniter
   * @return  miniter
   **/
  [[nodiscard]] size_t get_miniter() const { return miniter; }

  /**
   * @brief get residual method(default=0)
   * @return residual method number
   **/
  [[nodiscard]] size_t get_residual_method() const { return resid_method; }

  /**
   * @brief get print rhistory status
   * @return print rhistory true/false
   **/
  bool get_print_rhistory() const { return print_rhistory; }

  /**
   * @brief get handling scheme of initial vector handling
   * @return current handling scheme of initial vector
   */
  initvec_scheme get_initvec_scheme() const { return initvecscheme; }

  /**
   * @brief set the relaxation coefficient omega for SOR method (0 < w < 2,
   * Default: 1.9)
   * @param[in] w input omega value
   * @note
   * This variable is only used in SOR method
   */
  void set_omega(Float w) { omega = w; };

  /**
   * @brief get the relaxation coefficient omega for SOR method (Default: 1.9)
   * @return The relaxation coefficient of SOR
   * @note
   * This variable is only used in SOR method
   */
  Float get_omega() { return omega; };

  /**
   * @brief 0: no ordering 1: symrcm, 2: symamd, 3: csrmetisnd is used to reduce
   * zero fill-in.
   * @note
   * This variable is only used in sparse QR/Cholesky for GPU
   */
  void set_reorder(int r) { reorder = r; }

  /**
   * @brief 0: no ordering 1: symrcm, 2: symamd, 3: csrmetisnd is used to reduce
   * zero fill-in.
   * @note
   * This variable is only used in sparse QR/Cholesky for GPU
   */
  int get_reorder() { return reorder; }

  /**
   * @brief -1 if A is symmetric positive definite.
   * default reorder algorithm is csrmetisnd
   * @note
   * This variable is only used in sparse QR/Cholesky for GPU
   */
  int get_singularity() { return singularity; }

  double get_final_residual() { return final_resid; }
  size_t get_final_iter() { return final_iter; }
};

/**
 * @brief precondition base class
 */
template <typename MATRIX, typename Float> class precondition {
private:
public:
  vector<Float> M;
  MATRIX *A;

  std::function<void(MATRIX &)> create_precond;
  std::function<void(const vector<Float> &r, vector<Float> &z)> apply_precond;

  std::function<void(void)> get_precond();

  void set_precond_data(vector<Float> &m) { M = m; };
  vector<Float> get_precond_data() { return M; };

  precondition() {
    auto create = [](MATRIX &) {};
    auto apply = [](const vector<Float> &r, vector<Float> &z) { z = r; };
    create_precond = create;
    apply_precond = apply;
  };
};
/**@}*/
} // namespace solver
} // namespace monolish
