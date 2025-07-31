#ifndef __LAZYMATRIX_H__
#define __LAZYMATRIX_H__

#include <cassert>
#include <initializer_list>
#include <stdexcept>
#include <vector>

#ifdef __OPENMP
#include <omp.h>
#endif

namespace lm {

template <typename T, std::size_t Rows, std::size_t Cols> class Matrix {
public:
  using iterator = typename std::vector<T>::iterator;
  using const_iterator = typename std::vector<T>::const_iterator;

public:
  Matrix() { m_data.reserve(Rows * Cols); };

  Matrix(const std::vector<T> &data) : m_data(data) {
    if (data.size() != Rows * Cols) {
      throw std::runtime_error("Invalid matrix size");
    }
  }

  Matrix(std::initializer_list<std::initializer_list<T>> t_list) {
    if (t_list.size() != Rows) {
      throw std::invalid_argument("Invalid number of rows in initializer list");
    }

    m_data.resize(Rows * Cols); // Resizing the vector to the appropriate size

    for (const auto &row : t_list) {
      if (row.size() != Cols) {
        throw std::invalid_argument(
            "Invalid number of columns in initializer list");
      }
      for (const auto &elem : row) {
        m_data.emplace_back(elem);
      }
    }
  }

  constexpr auto operator()(std::size_t i, std::size_t j) const -> T {
    return m_data[i * Cols + j];
  }

  constexpr auto operator()(std::size_t i, std::size_t j) -> T & {
    return m_data[i * Cols + j];
  }

  constexpr auto rows() const -> std::size_t { return Rows; }
  constexpr auto cols() const -> std::size_t { return Cols; }

  constexpr auto begin() -> iterator { return m_data.begin(); }
  constexpr auto end() -> iterator { return m_data.end(); }
  constexpr auto cbegin() const -> const_iterator { return m_data.cbegin(); }
  constexpr auto cend() const -> const_iterator { return m_data.cend(); }

  template <typename Expr> auto operator=(const Expr &expr) -> Matrix & {
    for (std::size_t i = 0; i < Rows; i++) {
#ifdef __clang__
#pragma clang loop vectorize(enable)
#endif
      for (std::size_t j = 0; j < Cols; j++) {
        m_data[i * Cols + j] = expr(i, j);
      }
    }
    return *this;
  }

public:
private:
  std::vector<T> m_data;
};

// add different mat types

} // namespace lm

#endif // __LAZYMATRIX_H__
