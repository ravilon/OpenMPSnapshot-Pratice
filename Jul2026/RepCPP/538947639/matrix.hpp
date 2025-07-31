#pragma once

#include <vector>
#include <algorithm>
#include <ostream>
#include <iomanip>
#include <cassert>


// # Matrix #
// A class with basic functionality one would expect from a matrix
// - Stores data contiguously in a row-major order
//
/// >>> For all intended purposes this class is excessive (since we use raw arrays in the actual methods),
/// >>> but since there is no benefit in removing what is already done and works, I'll leave it as is.
template <typename T>
struct Matrix {
	// Initialize with value
	Matrix(size_t rows, size_t cols, T var = static_cast<T>(0)) : _rows(rows), _cols(cols), _data(rows * cols, var) {}
	Matrix(size_t size, T var = static_cast<T>(0)) : _rows(size), _cols(size), _data(size * size, var) {}

	// Initialize with data
	Matrix(size_t rows, size_t cols, T *data) : _rows(rows), _cols(cols), _data(data, data + rows * cols) {}
	Matrix(size_t size, T *data) : _rows(size), _cols(size), _data(data, data + size * size) {}
		// note the use of two iterator constructor 'vector(ptr, ptr + len)'

	// Copy constructor
	Matrix(const Matrix<T> &other) : _rows(other.rows()), _cols(other.cols()), _data(other._data) {}

	// Move constructor
	Matrix(Matrix<T> &&other) : _rows(other.rows()), _cols(other.cols()), _data(std::move(other._data)) {}

	// 2D indexation
	inline T& operator()(size_t i, size_t j) { return _data[i * _cols + j]; }
	inline const T& operator()(size_t i, size_t j) const { return _data[i * _cols + j]; }

	// 1D indexation
	inline T& operator[](size_t index) { return _data[index]; }
	inline const T& operator[](size_t index) const { return _data[index]; }

	// Getters
	size_t rows() const { return _rows; }
	size_t cols() const { return _cols; }
	T* data() { return _data.data(); }

	// Multiplication (naive implementation)
	Matrix<T> operator*(const Matrix<T> &other) {
		assert(this->cols() == other.rows() && "operator*(): incompatible matrices encountered.");

		Matrix<T> res(this->rows(), other.cols());

		for (size_t i = 0; i < this->rows(); ++i)
			for (size_t j = 0; j < other.cols(); ++j)
				for (size_t k = 0; k < this->cols(); ++k)
					res(i, j) += this->operator()(i, k) * other(k, j);
		return res;
	}

	// Addition
	Matrix<T> operator+(const Matrix<T> &other) {
		assert(this->rows() == other.rows() || this->cols() == other.cols()
			&& "operator+(): incompatible matrices encountered.");

		Matrix<T> res = *this;

		for (size_t i = 0; i < this->rows(); ++i)
			for (size_t j = 0; j < this->cols(); ++j)
				res(i, j) += other(i, j);
		return res;
	}

	// Substraction
	Matrix<T> operator-(const Matrix<T> &other) {
		assert(this->rows() == other.rows() || this->cols() == other.cols()
			&& "operator-(): incompatible matrices encountered.");

		Matrix<T> res = *this;

		for (size_t i = 0; i < this->rows(); ++i)
			for (size_t j = 0; j < this->cols(); ++j)
					res(i, j) -= other(i, j);
		return res;
	}

	// Utils
	Matrix<T>& randomize(T min = 0, T max = 100) {
		for (auto &elem : _data) elem = static_cast<T>(min + (max - min) * rand() / (RAND_MAX + 1.));
			// generate random double in [min, max] range and cast to 'T'

		for (size_t k = 0; k < std::min(_rows, _cols); ++k) this->operator()(k, k) *= 5;
			// loosely ensure maximum possible rank (no guarantees, but generally
			// having larger values on the main diagonal is good enough)

		return *this;
	}

	void downsize(size_t newRows, size_t newCols) {
		_rows = newRows;
		_cols = newCols;
		_data.resize(_rows * _cols);
	}

	T max_elem() const {
		T res = 0;

		for (const auto &el : _data)
			if (res < fabs(el)) res = fabs(el);

		return res;
	}

private:
	size_t _rows;
	size_t _cols;

public:
	std::vector<T> _data;
		// direct access for your dirty little needs, use with caution
};


// Typedefs
using LMatrix = Matrix<long double>;
using DMatrix = Matrix<double>;
using FMatrix = Matrix<float>;
using IMatrix = Matrix<int>;


// Printing matrices with std::cout
template<typename T>
std::ostream& operator<<(std::ostream &stream, const Matrix<T> &matrix) {
	constexpr size_t MAX_DISPLAYED_ROWS = 12;
	constexpr size_t MAX_DISPLAYED_COLS = 8;

	if (matrix.rows() <= MAX_DISPLAYED_ROWS && matrix.cols() <= MAX_DISPLAYED_COLS) {
		std::cout << '\n';

		for (size_t i = 0; i < matrix.rows(); ++i) {
			stream << std::setw(4) << '[';
			for (size_t j = 0; j < matrix.cols(); ++j) 
				if (std::abs(matrix(i, j)) < 1000) stream << std::setw(10) << std::setprecision(4) << std::defaultfloat << matrix(i, j);
				else stream << std::setw(10) << std::setprecision(1) << std::scientific << matrix(i, j);
			stream << std::setw(4) << ']' << '\n';
		}
	}
	else {
		stream << "<supressed>";
	}

	return stream;
}


// Verifies decomposition by computing (L * U - INITIAL_MATRIX)
// (here 'L' and 'U' are stored inside a single matrix 'A')
template <typename T>
T verify_LU(const Matrix<T> &A, const Matrix<T> &initial_matrix) {
	T err = 0;

	for (size_t i = 0; i < A.rows(); ++i)
		for (size_t j = 0; j < A.cols(); ++j) {
			T difference = initial_matrix(i, j);

			for (size_t k = 0; k <= std::min(j, i); ++k)
				difference -= (k == i) ? A(i, j) : A(i, k) * A(k, j);

			err = std::max(err, fabs(difference));
		}

	return err;
}