/**
 * Kei Imada
 * 20210120
 * Dense Vector implementation
 */

#pragma once

#include <fstream>
#include <iostream>
#include <string>

#include "matrix.h"

using namespace std;

/**
 * A dense vector of an arbitrary data type.
 * @tparam T the type of vector values
 */
template <typename T> class DenseVector : public Matrix<T> {
public:
  DenseVector(){};
  ~DenseVector();
  /**
   * Populates the vector from a matrix market filepath
   * @param matrix_market_filepath the filepath
   */
  void from_matrix_market_filepath(string const &matrix_market_filepath);
  /**
   * Populates the vector with zeros
   * @param num_zeros the number of zeros this vector has
   */
  void from_num_zeros(int num_zeros);
  /**
   * Populates the vector from another DenseVector
   * Basically a copy operation
   * @param vector the vector to copy from
   */
  void from_dense_vector(DenseVector<T> *vector);
  /**
   * Clears the vector to uninitialized state
   */
  void clear();
  /**
   * Prints the vector for debugging purposes
   */
  void print();
  /**
   * Checks if the vector is approximately equal to another vector
   * @param other_vector the vector to check approximate equality
   * @param threshold the order of difference compared to the original values
                      e.g. 1000 vs 1001 -> order of threshold = (1001 - 1000) / 1000 = 1/1000
     @return bool whether the other vectors is approximately equal to this vector
   */
  bool approx_equals(DenseVector<T> *other_vector, float threshold);

  // Getters

  int dimension_get() { return dimension; };
  T *values_get() { return values; };

private:
  int dimension = 0; // the number of elements in this vector
  T *values = nullptr;
};

template <typename T> DenseVector<T>::~DenseVector() { this->clear(); };

template <typename T>
void DenseVector<T>::from_matrix_market_filepath(
    string const &matrix_market_filepath) {
  ifstream file(matrix_market_filepath);

  // Ignore comments headers
  while (file.peek() == '%')
    file.ignore(2048, '\n');

  // Read number of rows and columns
  int num_col, num_lines;
  file >> dimension >> num_col >> num_lines;
  values = new T[dimension];
  // fill the matrix with data
  int cur_val_idx = 0;
  for (int l = 0; l < num_lines; l++) {
    T data;
    int row_idx, col;
    file >> row_idx >> col >> data;
    values[row_idx - 1] = data;
    while (cur_val_idx < dimension &&
           cur_val_idx < row_idx - 1) { // account for empty elements
      values[cur_val_idx++] = 0;
    }
    cur_val_idx = row_idx;
  }
  while (cur_val_idx < dimension) { // account for empty elements
    values[cur_val_idx++] = 0;
  }

  file.close();
};

template <typename T> void DenseVector<T>::from_num_zeros(int num_zeros) {
  dimension = num_zeros;
  values = new T[num_zeros];
  for (int i = 0; i < num_zeros; i++) {
    values[i] = 0;
  }
}

template <typename T>
void DenseVector<T>::from_dense_vector(DenseVector<T> *vector) {
  dimension = vector->dimension;
  values = new T[dimension];
  for (int i = 0; i < dimension; i++) {
    values[i] = vector->values_get()[i];
  }
}

template <typename T> void DenseVector<T>::clear() {
  delete[] values;
  values = nullptr;
  dimension = 0;
}

template <typename T>
bool DenseVector<T>::approx_equals(DenseVector<T> *other_vector, float threshold) {
  if (!other_vector || dimension != other_vector->dimension_get()) {
    return false;
  }
  for (int i = 0; i < dimension; i++) {
    if (abs(other_vector->values_get()[i] - values[i] + 0.0) /
            min(abs(other_vector->values_get()[i]), abs(values[i])) >
        threshold) {
      // if order of difference is larger than the threshold
      // e.g. 1000 vs 1001 -> order of threshold = (1001 - 1000) / 1000 = 1/1000
      return false;
    }
  }
  return true;
}

template <typename T> void DenseVector<T>::print() {
  cout << "DenseVector" << endl;
  cout << "  dimension: " << dimension << endl;
  cout << "  values:    ";
  for (int i = 0; i < dimension; i++) {
    if (values[i] != 0)
      cout << "(" << i << ", " << values[i] << ") ";
  }
  cout << endl;
  return;
};