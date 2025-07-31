#pragma once

#include <memory>
#include <cmath>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <stdio.h> // printf
#include <stdlib.h> // putenv
#include <limits> // std::numeric_limits<T>::max() and std::numeric_limits<T>::min();
#include <omp.h> // thread cancellation

#include "PRNG.h"

enum RandEnum { GAUSS, XAVIER, UNIFORM, NORMAL, LINEAR };
enum Operation { SUM, SUB, PROD, DIV };

// class pre-declaration for friend operator<< <>(...)
template<typename T>
class Matrix;

template<typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& matrix);

template<typename T>
class Matrix {
public:
	Matrix(int num_threads = 1);
	Matrix(int rows, int cols, T value = 0, int num_threads = 1);
	template<typename U>
	Matrix(int rows, int cols, RandEnum distr, T param1, U param2, int num_threads = 1);
	template<typename U>
	Matrix(int rows, int cols, RandEnum distr, const T* param_vect1, const U* param_vect2, int num_threads = 1);
	template<typename U>
	Matrix(int rows, int cols, RandEnum distr, Matrix<T> param_matrix1, Matrix<U> param_matrix2, int num_threads = 1);
	Matrix(const T* array, int rows, int cols, int num_threads = 1);
	// TEMPORARY constructor -> extend to variadic constructor
	Matrix(const T* arr1, const T* arr2, int rows, int cols, int num_threads=1);
	//Matrix(const T*... arrays); // TODO construct matrix from variadic number of arrays
	//Matrix(std::initializer_list<const T* array> arrays);

	Matrix(Matrix<T>&& other) noexcept;
	Matrix(const Matrix<T>& other);

	// ** getters **
	int getMemory();
	int getThreadsN();
	int getRows() const;
	int getCols() const;
	Matrix<T> getSlice(int from_i, int to_i, int from_j, int to_j);
	Matrix<T> row(int i);
	Matrix<T> col(int j);

	// ** setters **
	void setThreads(int n_threads);

	// ** methods **
	void applyFunc(void (*userFunc)(T&));
	Matrix<T> transpose();
	Matrix<T> dot(const Matrix<T>& other);
	Matrix<T> dot_v2(const Matrix<T>& other);
	Matrix<T> dotTranspose(const Matrix<T>& rhs);
	Matrix<T> hSum();
	Matrix<T> vSum();
	Matrix<T>& hBroadcast(const Matrix<T>& filter, Operation op);
	Matrix<T>& vBroadcast(const Matrix<T>& filter, Operation op);
	Matrix<T>& vShuffle();
	Matrix<T>& insert(const Matrix<T>& rhs, int from_i, int to_i, int from_j, int to_j);
	/** TODO: */
	Matrix<T>& hStack(const T* array, int rows_ext, int cols_ext);
	Matrix<T>& hStack(const Matrix<T>& other);
	//Matrix<T>& hStack(Matrix<T>&& other);
	Matrix<T>& vStack(const T* array, int rows_ext, int cols_ext);
	Matrix<T>& vStack(const Matrix<T>& other);
	//Matrix<T>& vStack(Matrix<T>&& other);

	Matrix<int> hMinIndex();
	Matrix<int> hMaxIndex();
	Matrix<int> vMinIndex();
	Matrix<int> vMaxIndex();
	std::pair<int, int> minIndex();
	std::pair<int, int> maxIndex();
	Matrix<T> hMin();
	Matrix<T> hMax();
	Matrix<T> vMin();
	Matrix<T> vMax();
	T min();
	T max();
	
	bool isEqual(const Matrix<T>& rhs);
	// ** operators **
	T& operator()(const int& row, const int& col);
	const T& operator()(const int& row, const int& col) const;
	Matrix<T>& operator=(const Matrix<T>& other);

	// ** math. operations **
	// matrix -- matrix
	Matrix<T> operator+(const Matrix<T>& other);
	Matrix<T>& operator+=(const Matrix<T>& other);
	Matrix<T> operator-(const Matrix<T>& other);
	Matrix<T>& operator-=(const Matrix<T>& other);
	Matrix<T> operator*(const Matrix<T>& other);
	Matrix<T>& operator*=(const Matrix<T>& other);
	// matrix -- scalar (rhs)
	Matrix<T> operator+(T scalar);
	Matrix<T> operator-(T scalar);
	Matrix<T> operator*(T scalar);
	Matrix<T> operator/(T scalar);
	Matrix<T>& operator+=(T scalar);
	Matrix<T>& operator*=(T scalar);
	Matrix<T>& operator-=(T scalar);
	Matrix<T>& operator/=(T scalar);

	// ** visualization methods **
	void print(std::ostream& STREAM) const;
	
	// ** iterators **
	T* begin() { return _matrix.get(); }
	T* end() { return begin() + _rows*_cols; }
    	const T* begin() const { return _matrix.get(); }
	const T* end() const { return begin() + _rows*_cols; }
	T* rowBegin(int row) { return begin() + row*_cols; }
	T* rowEnd(int row) { return rowBegin(row) + _cols; }
	const T* rowBegin(int row) const { return begin() + row*_cols; }
	const T* rowEnd(int row) const { return rowBegin(row) + _cols; }

protected:
	int _n_threads = 1;
	bool _threads_enabled = false;

	int _rows;
	int _cols;
	std::unique_ptr<T[]> _matrix;

	friend std::ostream& operator<< <>(std::ostream& out, const Matrix<T>& matrix);
       
	};
//////////////////////////////////////////////////////////////////////
			/// (SCALAR -- MATRIX) MATH. OPERATIONS (LHS) ///
//////////////////////////////////////////////////////////////////////

template<typename T, typename U>
Matrix<T> operator+(U scalar, Matrix<T> rhs) {
	T new_scalar = static_cast<T>(scalar);
	return rhs += new_scalar;
}

template<typename T, typename U>
Matrix<T> operator*(U scalar, Matrix<T> rhs) {
	T new_scalar = static_cast<T>(scalar);
	return rhs *= new_scalar;
}

template<typename T, typename U>
Matrix<T> operator-(U scalar, Matrix<T> rhs) {
	T new_scalar = static_cast<T>(scalar);
	return new_scalar + (-rhs);
}

template<typename T>
Matrix<T> operator-(Matrix<T> rhs) {
	return rhs *= (-1);
}

//////////////////////////////////////////////////////////////////////
					/// NON MEMBER FUNCTIONS ///
//////////////////////////////////////////////////////////////////////

template<typename T>
std::ostream& operator<<(std::ostream& out, const Matrix<T>& matrix) {
	matrix.print(out);
	return out;
}

//////////////////////////////////////////////////////////////////////
//																	//
//							 IMPLEMENTATION							//
//																	//
//////////////////////////////////////////////////////////////////////

template<typename T>
Matrix<T>::Matrix(int num_threads) : 
	_n_threads{ num_threads }{
	
	_rows = 0;
	_cols = 0;
	if (num_threads > 1) { _threads_enabled = true; }
}

template<typename T>
Matrix<T>::Matrix(int rows, int cols, T value, int num_threads) : 
	_rows{ rows }, _cols{ cols },
	_n_threads{ num_threads },
	_matrix{ std::make_unique<T[]>(rows*cols) } {

	if (num_threads > 1) { _threads_enabled = true; }

	#pragma omp parallel for num_threads(_n_threads) if(_threads_enabled)
	for (int n = 0; n < _rows*_cols; ++n) {
		_matrix[n] = value;
	}
}

/*
 * param1, param2: depends on distr
					if distr UNIFORM: range [param1, param2]
					if distr GAUSS: mean = param1, std = param2
					if distr XAVIER: mean = 0, std = sqrt(2 / (param1 + param2))
					if distr LINEAR: min, max
*/
template<typename T>
template<typename U>
Matrix<T>::Matrix(int rows, int cols, RandEnum distr, T param1, U param2, int num_threads) :
	_rows{ rows }, _cols{ cols },
	_n_threads{ num_threads },
	_matrix{ std::make_unique<T[]>(rows*cols) } {
	/**
	 * TODO: normal_dist<T>() and uniform_dist<T>()
	 * are currently NOT THREAD SAFE
	*/

	if (num_threads > 1) { _threads_enabled = true; }

	switch (distr) {
		case UNIFORM:{
			uniform_dist<T> r1(param1, param2);
			#pragma omp parallel for simd num_threads(_n_threads) if(_threads_enabled)
			for (int n = 0; n < _rows*_cols; ++n) {
				_matrix[n] = r1();
			}
			break;
		}
		case GAUSS:{
			normal_dist<T> r2(param1, param2);
			#pragma omp parallel for simd num_threads(_n_threads) if(_threads_enabled)
			for (int n = 0; n < _rows*_cols; ++n) {
				_matrix[n] = r2();
			}
			break;
		}
		case NORMAL:{
			normal_dist<T> r3(param1, param2);
			#pragma omp parallel for simd num_threads(_n_threads) if(_threads_enabled)
			for (int n = 0; n < _rows*_cols; ++n) {
				_matrix[n] = r3();
			}
			break;
		}
		case XAVIER:{
			normal_dist<T> r4(0, sqrt(2/(param1+param2)));
			#pragma omp parallel for simd num_threads(_n_threads) if(_threads_enabled)
			for (int n = 0; n < _rows*_cols; ++n) {
				_matrix[n] = r4();
			}
			break;
		}
		case LINEAR:{
			float cst = _cols-1;
			#pragma omp parallel for simd collapse(2) num_threads(_n_threads) if(_threads_enabled)
			for(int i = 0; i < _rows; ++i){
				for(int j = 0; j < _cols; ++j){
					_matrix[j+i*_cols] = static_cast<T>(param2+(_cols-j-1)/cst*(param1-param2));
				}
			}
			break;
		}
		default:{
			std::cout << "Distribution not specified. Using: UNIFORM" << std::endl;
			uniform_dist<T> r5(param1, param2);
			#pragma omp parallel for simd num_threads(1) if(_threads_enabled)
			for (int n = 0; n < _rows*_cols; ++n) {
				_matrix[n] = r5();
			}
			break;
		}
	};
}

template<typename T>
template<typename U>
Matrix<T>::Matrix(int rows, int cols, RandEnum distr, const T* param_vect1, const U* param_vect2, int num_threads) :
	_rows{ rows }, _cols{ cols },
	_n_threads{ num_threads },
	_matrix{ std::make_unique<T[]>(rows*cols) }{
	
	if (num_threads > 1) { _threads_enabled = true; }

	switch (distr) {
		case UNIFORM:{
			for(int i = 0; i < _rows; ++i){
				uniform_dist<T> r1(param_vect1[i], param_vect2[i]);
				#pragma omp parallel for simd num_threads(_n_threads) if(_threads_enabled)
				for (int j = 0; j < _cols; ++j) {
					_matrix[j+i*_cols] = r1();
				}
			}
			break;
		}
		case GAUSS:{
			for(int i = 0; i < _rows; ++i){
				normal_dist<T> r2(param_vect1[i], param_vect2[i]);
				#pragma omp parallel for simd num_threads(_n_threads) if(_threads_enabled)
				for (int j = 0; j < _cols; ++j) {
					_matrix[j+i*_cols] = r2();
				}
			}
			break;
		}
		case NORMAL:{
			for(int i = 0; i < _rows; ++i){
				normal_dist<T> r3(param_vect1[i], param_vect2[i]);
				#pragma omp parallel for simd num_threads(_n_threads) if(_threads_enabled)
				for (int j = 0; j < _cols; ++j) {
					_matrix[j+i*_cols] = r3();
				}
			}
			break;
		}
		case XAVIER:{
			for(int i = 0; i < _rows; ++i){
				normal_dist<T> r4(0, sqrt(2/(param_vect1[i]+param_vect2[i])));
				#pragma omp parallel for simd num_threads(_n_threads) if(_threads_enabled)
				for (int j = 0; j < _cols; ++j) {
					_matrix[j+i*_cols] = r4();
				}
			}
			break;
		}
		case LINEAR:{
			float cst = _cols-1;
			#pragma omp parallel for collapse(1) num_threads(_n_threads) if(_threads_enabled)
			for(int i = 0; i < _rows; ++i){
				T inf = param_vect1[i];
				T sup = param_vect2[i];
				#pragma omp simd
				for(int j = 0; j < _cols; ++j){
					_matrix[j+i*_cols] = static_cast<T>(sup+(_cols-j-1)/cst*(inf-sup));
				}
			}
			break;
		}
		default:{
			std::cout << "Distribution not specified." << std::endl;
			break;
		}
	};
}

template<typename T>
template<typename U>
Matrix<T>::Matrix(int rows, int cols, RandEnum distr, Matrix<T> param_matrix1, Matrix<U> param_matrix2, int num_threads) :
	_rows{ rows }, _cols{ cols },
	_n_threads{ num_threads },
	_matrix{ std::make_unique<T[]>(rows*cols) }{

	if (num_threads > 1) { _threads_enabled = true; }

	if((param_matrix2.getCols() != param_matrix1.getCols()) || (param_matrix2.getRows() != param_matrix1.getRows())){
		printf("Error: parametric matrices must be of same dimensions\n");
	}
	else if(param_matrix1.getCols() == 1){
		param_matrix1 = param_matrix1.transpose();
		param_matrix2 = param_matrix2.transpose();
	}

	switch (distr) {
		case UNIFORM:{
			for(int i = 0; i < _rows; ++i){
				uniform_dist<T> r1(param_matrix1(0, i), param_matrix2(0, i));
				#pragma omp parallel for simd num_threads(_n_threads) if(_threads_enabled)
				for (int j = 0; j < _cols; ++j) {
					_matrix[j+i*_cols] = r1();
				}
			}
			break;
		}
		case GAUSS:{
			for(int i = 0; i < _rows; ++i){
				normal_dist<T> r2(param_matrix1(0, i), param_matrix2(0, i));
				#pragma omp parallel for simd num_threads(_n_threads) if(_threads_enabled)
				for (int j = 0; j < _cols; ++j) {
					_matrix[j+i*_cols] = r2();
				}
			}
			break;
		}
		case NORMAL:{
			for(int i = 0; i < _rows; ++i){
				normal_dist<T> r3(param_matrix1(0, i), param_matrix2(0, i));
				#pragma omp parallel for simd num_threads(_n_threads) if(_threads_enabled)
				for (int j = 0; j < _cols; ++j) {
					_matrix[j+i*_cols] = r3();
				}
			}
			break;
		}
		case XAVIER:{
			for(int i = 0; i < _rows; ++i){
				normal_dist<T> r4(0, sqrt(2/(param_matrix1(0, i) + param_matrix2(0, i))));
				#pragma omp parallel for simd num_threads(_n_threads) if(_threads_enabled)
				for (int j = 0; j < _cols; ++j) {
					_matrix[j+i*_cols] = r4();
				}
			}
			break;
		}
		case LINEAR:{
			float cst = _cols-1;
			#pragma omp parallel for collapse(1) num_threads(_n_threads) if(_threads_enabled)
			for(int i = 0; i < _rows; ++i){
				T inf = param_matrix1(0, i);
				T sup = param_matrix2(0, i);
				#pragma omp simd
				for(int j = 0; j < _cols; ++j){
					_matrix[j+i*_cols] = static_cast<T>(sup+(_cols-j-1)/cst*(inf-sup));
				}
			}
			break;
		}
		default:{
			std::cout << "Distribution not specified." << std::endl;
			break;
		}
	};
}

template<typename T>
Matrix<T>::Matrix(const T* array, int rows, int cols, int num_threads) : 
	_rows{ rows }, _cols{ cols },
	_n_threads{ num_threads },
	_matrix{ std::make_unique<T[]>(rows*cols) } {
	
	if (num_threads > 1) { _threads_enabled = true; }
	#pragma omp parallel for num_threads(_n_threads) if(_threads_enabled)
	for(int i = 0; i < _rows; ++i){
		#pragma omp simd
		for(int j = 0; j < _cols; ++j){
			_matrix[j+i*_cols] = array[j+i*_cols];
		}
	}
}

/**
 * TODO: THIS IS A TEMPORARY CONSTRUCTOR
 * NEED TO IMPLEMENT VARIADIC ONE
*/
template<typename T>
Matrix<T>::Matrix(const T* arr1, const T* arr2, int rows, int cols, int num_threads) : 
	_rows{ rows }, _cols{ cols }, 
	_n_threads{ num_threads },
	_matrix{ std::make_unique<T[]>(rows*cols) } {

	if (num_threads > 1) { _threads_enabled = true; }
	
	#pragma omp parallel for simd num_threads(_n_threads) if(_threads_enabled)
	for(int j = 0; j < _cols; ++j){
		_matrix[j] = arr1[j]; // j+0*_cols
	}
	#pragma omp parallel for simd num_threads(_n_threads) if(_threads_enabled)
	for(int j = 0; j < _cols; ++j){
		_matrix[j+_cols] = arr2[j]; // j+1*_cols
	}
}

template<typename T>
Matrix<T>::Matrix(Matrix<T>&& other) noexcept: 
	_matrix{ std::move(other._matrix) }, 
	_rows{ other._rows }, _cols{ other._cols },
	_n_threads{ other._n_threads },
	_threads_enabled{ other._threads_enabled } {
}

template<typename T>
Matrix<T>::Matrix(const Matrix<T>& other) { 
	if (other._n_threads > 1) { 
		_threads_enabled = true;
		_n_threads = other._n_threads;
	}
	*this = other; 
}

//////////////////////////////////////////////////////////////////////
						/// GETTERS ///
//////////////////////////////////////////////////////////////////////

template<typename T>
int Matrix<T>::getMemory() { return static_cast<size_t>(_rows*_cols*sizeof(T)); }
template<typename T>
int Matrix<T>::getThreadsN() { return _n_threads; }
template<typename T>
int Matrix<T>::getRows() const { return _rows; }
template<typename T>
int Matrix<T>::getCols() const { return _cols; }
/**
 * Slice of this._matrix to extract
 * range: 
 * 		[from_i, to_i)
 * 		[from_j, to_j)
*/
template<typename T>
Matrix<T> Matrix<T>::getSlice(int from_i, int to_i, int from_j, int to_j){
	int new_rows = to_i-from_i;
	int new_cols = to_j-from_j;
	Matrix<T> slice(new_rows, new_cols, 0, _n_threads);
	#pragma omp parallel for simd collapse(2) num_threads(_n_threads) if(_threads_enabled)
	for(int i = 0; i < new_rows; ++i){
		for(int j = 0; j < new_cols; ++j){
			slice(i, j) = _matrix[(from_j+j)+_cols*(from_i+i)];
		}
	}
	return slice;
}

template<typename T>
Matrix<T> Matrix<T>::row(int i){
	return this->getSlice(i, i+1, 0, _cols);
}

template<typename T>
Matrix<T> Matrix<T>::col(int j){
	return this->getSlice(0, _rows, j, j+1);
}

//////////////////////////////////////////////////////////////////////
						/// SETTERS ///
//////////////////////////////////////////////////////////////////////

template<typename T>
void Matrix<T>::setThreads(int n_threads) { 
	_n_threads = n_threads;
	if(_n_threads > 1){ _threads_enabled = true; } 
	else { _threads_enabled = false; }
}

//////////////////////////////////////////////////////////////////////
							/// METHODS ///
//////////////////////////////////////////////////////////////////////

template<typename T>
void Matrix<T>::applyFunc(void (*userFunc)(T&)){
	#pragma omp parallel for simd collapse(2) num_threads(_n_threads) if(_threads_enabled)
	for(int i = 0; i < _rows; ++i){
		for(int j = 0; j < _cols; ++j){
			userFunc(_matrix[j+i*_cols]);
		}
	}
}

template<typename T>
Matrix<T> Matrix<T>::transpose(){
	Matrix<T> res(_cols, _rows, 0, _n_threads);
	#pragma omp parallel for collapse(2) num_threads(_n_threads) if(_threads_enabled) 
	for(int i = 0; i < _rows; ++i){
		for(int j = 0; j < _cols; ++j){
			res(j, i) = _matrix[j+i*_cols];
		}
	}
	return res;
}

template<typename T>
Matrix<T> Matrix<T>::dot(const Matrix<T>& other) {
	assert(other.getRows() == _cols);

	int new_cols = other.getCols();
	Matrix<T> resultMatrix(_rows, new_cols, 0, _n_threads);
	
	#pragma omp parallel for collapse(2) num_threads(_n_threads) if(_threads_enabled)
	for(int i = 0; i < _rows; ++i){
		for(int j = 0; j < new_cols; ++j){
			T reduc_scalar = 0;
			#pragma omp simd reduction(+:reduc_scalar)
			for(int k = 0; k < _cols; ++k){
				reduc_scalar += _matrix[k+i*_cols] * other(k, j);
			}
			resultMatrix(i, j) = reduc_scalar;
		}
	}
	return resultMatrix;
}
/**
 * Use only if you are certain that it will be faster than the regular dot()
 * It happened to be faster when other is big and other.getCols()>other.getRows() (tested with 1 to 4 threads)
*/
template<typename T>
Matrix<T> Matrix<T>::dot_v2(const Matrix<T>& other) {
	assert(other.getRows() == _cols);

	int new_cols = other.getCols();
	Matrix<T> resultMatrix(_rows, new_cols, 0, _n_threads);
	
	const int buff_len = new_cols;
	T shared_buff[buff_len];
	#pragma omp parallel for num_threads(_n_threads) if(_threads_enabled)
	for(int j = 0; j < new_cols; ++j){
			shared_buff[j] = 0;
	}
	for(int i = 0; i < _rows; ++i){
		#pragma omp parallel for collapse(2) shared(shared_buff) num_threads(_n_threads) if(_threads_enabled)
		for(int k = 0; k < _cols; ++k){
			for(int j = 0; j < new_cols; ++j){
				#pragma omp atomic
				shared_buff[j] += _matrix[k+i*_cols] * other(k, j);
			}
		}
		#pragma omp parallel for num_threads(_n_threads) if(_threads_enabled)
		for(int j = 0; j < new_cols; ++j){
			resultMatrix(i, j) = shared_buff[j];
			shared_buff[j] = 0;
		}
	}
	return resultMatrix;
}

/**
 * Performs C = A.dot(B.T) without doing B.T explicitly
 * Advantages: 
 * 		- we skip the B.T step
 * 		- we take advantage of memory coalescing in both matrices 
 * 		instead of just the left one (as in a standard (obj1).dot(obj2) op)
*/
template<typename T>
Matrix<T> Matrix<T>::dotTranspose(const Matrix<T>& rhs){
	int new_rows = _rows;
	int new_cols = rhs.getRows();
	Matrix res(new_rows, new_cols, 0, _n_threads);

	#pragma omp parallel for collapse(2) num_threads(_n_threads) if(_threads_enabled)
	for(int i = 0; i < new_rows; ++i){
		for(int j = 0; j < new_cols; ++j){
			T reduc_scalar = 0;
			#pragma omp simd reduction(+:reduc_scalar)
			for(int k = 0; k < _cols; ++k){
				reduc_scalar += _matrix[k+i*_cols] * rhs(j, k);
			}
			res(i, j) = reduc_scalar;
		}
	}
	return res;
}

/*
 * [[., ., ., .],    [[+],
 *  [., ., ., .], ->  [+],
 *  [., ., ., .]]     [+]]
 */
template<typename T>
Matrix<T> Matrix<T>::hSum(){
	Matrix<T> res(_rows, 1, 0, _n_threads);
	
	#pragma omp parallel for num_threads(_n_threads) if(_threads_enabled)
	for(int i = 0; i < _rows; ++i){
		T reduc_scalar = 0;
		#pragma omp simd reduction(+:reduc_scalar)
		for(int j = 0; j < _cols; ++j){
			reduc_scalar += _matrix[j+i*_cols];
		}
		res(i, 0) = reduc_scalar;
	}
	return res;
}
/*
 * [[., ., ., .],
 *  [., ., ., .], -> [[+, +, +, +]]
 *  [., ., ., .]]
 */
template<typename T>
Matrix<T> Matrix<T>::vSum(){
	/**
	 * TODO: reduction seems faster when cols >> rows
	 * this method needs tests to check which method 
	 * between reduction and atomic needs to be performed
	 * w.r.t. the different threads numbers
	 * 
	 * atomic is generally faster though. 
	 * Very few memory coallescing here ; 
	 * this is why when cols >> rows -> more 
	 * coallescing mem -> reduction faster
	*/
	Matrix<T> res(1, _cols, 0, _n_threads);
	/*
	#pragma omp parallel for collapse(1) num_threads(_n_threads) if(_threads_enabled)
	for(int j = 0; j < _cols; ++j){
		T reduc_scalar = 0;
		#pragma omp simd reduction(+:reduc_scalar)
		for(int i = 0; i < _rows; ++i){
			reduc_scalar += _matrix[j+i*_cols];
		}
		res(0, j) = reduc_scalar;
	}
	*/
	#pragma omp parallel for collapse(2) num_threads(_n_threads) if(_threads_enabled)
	for(int i = 0; i < _rows; ++i){
		for(int j = 0; j < _cols; ++j){
			#pragma omp atomic
			res(0, j) += _matrix[j+i*_cols];
		}
	}
	return res;
}
/*
 filter must be a Matrix<T> with one of its dimensions being one
 * [[0, 1, 2, 3],     [[ 2],     [[0 * 2, 1 * 2, 2 * 2, 3 * 2],
 *  [2,-1,-2, 1]]  *   [-3]]  ->  [2 *-3,-1 *-3,-2 *-3, 1 *-3],
 */
template<typename T>
Matrix<T>& Matrix<T>::hBroadcast(const Matrix<T>& filter, Operation op){
	
	switch (op) {
		case SUM:{
			#pragma omp parallel for num_threads(_n_threads) if(_threads_enabled)
			for(int i = 0; i < _rows; ++i){
				#pragma omp simd
				for(int j = 0; j < _cols; ++j){
					_matrix[j+i*_cols] += filter(i, 0);
				}
			}
			break;
		}
		case SUB:{
			#pragma omp parallel for num_threads(_n_threads) if(_threads_enabled)
			for(int i = 0; i < _rows; ++i){
				#pragma omp simd
				for(int j = 0; j < _cols; ++j){
					_matrix[j+i*_cols] -= filter(i, 0);
				}
			}
			break;
		}
		case PROD:{
			#pragma omp parallel for num_threads(_n_threads) if(_threads_enabled)
			for(int i = 0; i < _rows; ++i){
				#pragma omp simd
				for(int j = 0; j < _cols; ++j){
					_matrix[j+i*_cols] *= filter(i, 0);
				}
			}
			break;
		}
		case DIV:{
			#pragma omp parallel for num_threads(_n_threads) if(_threads_enabled)
			for(int i = 0; i < _rows; ++i){
				#pragma omp simd
				for(int j = 0; j < _cols; ++j){
					_matrix[j+i*_cols] /= filter(i, 0);
				}
			}
			break;
		}
		default:{
			printf("No broadcast op has been performed...\n");
			break;
		}
	};
	return *this;
}
/*
 * filter must be a Matrix<T> with one of its dimensions being one
 * [[0, 1, 2, 3],                        [[0 * 4, 1 * 5, 2 * 3, 3 * 1],
 *  [2,-1,-2, 1]]  *  [[4, 5, 3, 1]]  ->  [2 * 4,-1 * 5,-2 * 3, 1 * 1],
 */
template<typename T>
Matrix<T>& Matrix<T>::vBroadcast(const Matrix<T>& filter, Operation op){
	switch (op) {
		case SUM:{
			#pragma omp parallel for num_threads(_n_threads) if(_threads_enabled)
			for(int j = 0; j < _cols; ++j){
				#pragma omp simd
				for(int i = 0; i < _rows; ++i){
					_matrix[j+i*_cols] += filter(0, j);
				}
			}
			break;
		}
		case SUB:{
			#pragma omp parallel for num_threads(_n_threads) if(_threads_enabled)
			for(int j = 0; j < _cols; ++j){
				#pragma omp simd
				for(int i = 0; i < _rows; ++i){
					_matrix[j+i*_cols] -= filter(0, j);
				}
			}
			break;
		}
		case PROD:{
			#pragma omp parallel for num_threads(_n_threads) if(_threads_enabled)
			for(int j = 0; j < _cols; ++j){
				#pragma omp simd
				for(int i = 0; i < _rows; ++i){
					_matrix[j+i*_cols] *= filter(0, j);
				}
			}
			break;
		}
		case DIV:{
			#pragma omp parallel for num_threads(_n_threads) if(_threads_enabled)
			for(int j = 0; j < _cols; ++j){
				#pragma omp simd
				for(int i = 0; i < _rows; ++i){
					_matrix[j+i*_cols] /= filter(0, j);
				}
			}
			break;
		}
		default:{
			printf("No broadcast op has been performed...\n");
			break;
		}
	};
	return *this;
}
/**
 * [[0, 1, 2, 3], (0)                          [[4, 5, 6, 7], (1)
 *  [4, 5, 6, 7], (1) we shuffle indices... ->  [8, 9, 0, 1], (2)
 *  [8, 9, 0, 1]] (2)                           [0, 1, 2, 3]] (0)
 */
template<typename T>
Matrix<T>& Matrix<T>::vShuffle(){
	std::vector<int> indices;
	indices.reserve(_rows);
	#pragma omp parallel for num_threads(_n_threads) if(_threads_enabled)
	for(int i = 0; i < _rows; ++i){
		indices.push_back(i);
	}
	std::random_shuffle(indices.begin(), indices.end());
	
	Matrix<T> temp(_rows, _cols, 0, _n_threads);
	#pragma omp parallel for collapse(2) num_threads(_n_threads) if(_threads_enabled)
	for(int i = 0; i < _rows; ++i){
		for(int j = 0; j < _cols; ++j){
			int rand_i = indices[i];
			temp(i, j) = _matrix[j+rand_i*_cols];
		}
	}
	(*this) = temp;
	return *this;
}
template<typename T>
Matrix<T>& Matrix<T>::insert(const Matrix<T>& rhs, int from_i, int to_i, int from_j, int to_j){
	#pragma omp parallel for simd collapse(2) num_threads(_n_threads) if(_threads_enabled)
	for(int i = from_i; i < to_i; ++i){
		for(int j = from_j; j < to_j; ++j){
			_matrix[j+_cols*i] = rhs(i-from_i, j-from_j);
		}
	}
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::hStack(const T* array, int rows_ext, int cols_ext){
	std::unique_ptr<T[]> temp = std::make_unique<T[]>(_rows*(_cols+cols_ext));
	#pragma omp parallel for num_threads(_n_threads) if(_threads_enabled)
	for(int i = 0; i < _rows; ++i){
		#pragma omp simd
		for(int j = 0; j < _cols; ++j){
			temp[j+i*(_cols+cols_ext)] = _matrix[j+i*_cols];
		}
		#pragma omp simd
		for(int j = _cols; j < _cols+cols_ext; ++j){
			temp[j+i*(_cols+cols_ext)] = array[(j-_cols)+i*cols_ext];
		}
	}
	_matrix = std::move(temp);
	_cols += cols_ext;
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::hStack(const Matrix<T>& other){
	std::unique_ptr<T[]> temp = std::make_unique<T[]>(_rows*(_cols+other.getCols()));
	#pragma omp parallel for num_threads(_n_threads) if(_threads_enabled)
	for(int i = 0; i < _rows; ++i){
		#pragma omp simd
		for(int j = 0; j < _cols; ++j){
			temp[j+i*(_cols+other.getCols())] = _matrix[j+i*_cols];
		}
		#pragma omp simd
		for(int j = _cols; j < _cols+other.getCols(); ++j){
			temp[j+i*(_cols+other.getCols())] = other(i, j-_cols);
		}
	}
	_matrix = std::move(temp);
	_cols += other.getCols();
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::vStack(const T* array, int rows_ext, int cols_ext){
	std::unique_ptr<T[]> temp = std::make_unique<T[]>((_rows+rows_ext)*_cols);
	#pragma omp parallel for simd collapse(2) num_threads(_n_threads) if(_threads_enabled)
	for(int i = 0; i < _rows; ++i){
		for(int j = 0; j < _cols; ++j){
			temp[j+i*_cols] = _matrix[j+i*_cols];
		}
	}
	#pragma omp parallel for simd collapse(2) num_threads(_n_threads) if(_threads_enabled)
	for(int i = _rows; i < _rows+rows_ext; ++i){
		for(int j = 0; j < _cols; ++j){
			temp[j+i*_cols] = array[j+(i-_rows)*_cols];
		}
	}
	_matrix = std::move(temp);
	_rows += rows_ext;
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::vStack(const Matrix<T>& other){
	std::unique_ptr<T[]> temp = std::make_unique<T[]>((_rows+other.getRows())*_cols);
	#pragma omp parallel for simd collapse(2) num_threads(_n_threads) if(_threads_enabled)
	for(int i = 0; i < _rows; ++i){
		for(int j = 0; j < _cols; ++j){
			temp[j+i*_cols] = _matrix[j+i*_cols];
		}
	}
	#pragma omp parallel for simd collapse(2) num_threads(_n_threads) if(_threads_enabled)
	for(int i = _rows; i < _rows+other.getRows(); ++i){
		for(int j = 0; j < _cols; ++j){
			temp[j+i*_cols] = other(i-_rows, j);
		}
	}
	_matrix = std::move(temp);
	_rows += other.getRows();
	return *this;
}
/**
 * Returns a horizontal vector matrix containing
 * the row index of each column-wise min
 * 
 * [[0, 1, 2, 3],
 *  [4, 5, 6, 7],  ->  [0, 0, 0, 2]
 *  [8, 9, 0, 1]]
 */
template<typename T>
Matrix<int> Matrix<T>::hMinIndex(){
	T curr_min = std::numeric_limits<T>::max();
	Matrix<T> minBuff(1, _cols, curr_min, _n_threads);
	Matrix<int> minIdxBuff(1, _cols, -1, _n_threads);
	/**
	 * would for(j...){for(i...)} order make it possible to remove atomic write?
	*/
	#pragma omp parallel for collapse(2) num_threads(_n_threads) if(_threads_enabled)
	for(int i = 0; i < _rows; ++i){
		for(int j = 0; j < _cols; ++j){
			T& inner_curr_min = minBuff(0, j);
			const T& curr_val = _matrix[j+i*_cols];
			if(curr_val < inner_curr_min){ 
				#pragma atomic write
				inner_curr_min = curr_val;
				#pragma atomic write
				minIdxBuff(0, j) = i;
			}
		}
	}
	return minIdxBuff;
}
/**
 * Returns a horizontal vector matrix containing 
 * the row index of each column-wise max
 * 
 * [[0, 1, 2, 3],
 *  [4, 5, 6, 7],  ->  [2, 2, 1, 1]
 *  [8, 9, 0, 1]]
 */
template<typename T>
Matrix<int> Matrix<T>::hMaxIndex(){
	T curr_max = std::numeric_limits<T>::min();
	Matrix<T> maxBuff(1, _cols, curr_max, _n_threads);
	Matrix<int> maxIdxBuff(1, _cols, -1, _n_threads);
	/**
	 * would for(j...){for(i...)} order make it possible to remove atomic write?
	*/
	#pragma omp parallel for collapse(2) num_threads(_n_threads) if(_threads_enabled)
	for(int i = 0; i < _rows; ++i){
		for(int j = 0; j < _cols; ++j){
			T& inner_curr_max = maxBuff(0, j);
			const T& curr_val = _matrix[j+i*_cols];
			if(curr_val > inner_curr_max){ 
				#pragma atomic write
				inner_curr_max = curr_val;
				#pragma atomic write
				maxIdxBuff(0, j) = i;
			}
		}
	}
	return maxIdxBuff;
}
/**
 * Returns a vertical vector matrix containing 
 * the column index of each row-wise min
 * 
 * [[0, 1, 2, 3],	   [[0],
 *  [4, 5, 6, 7],  ->   [0],
 *  [8, 9, 0, 1]]		[2]]
 */
template<typename T>
Matrix<int> Matrix<T>::vMinIndex(){
	T curr_min = std::numeric_limits<T>::max();
	Matrix<T> minBuff(_rows, 1, curr_min, _n_threads);
	Matrix<int> minIdxBuff(_rows, 1, -1, _n_threads);

	#pragma omp parallel for collapse(1) num_threads(_n_threads) if(_threads_enabled)
	for(int i = 0; i < _rows; ++i){
		T& inner_curr_min = minBuff(i, 0);
		#pragma omp simd
		for(int j = 0; j < _cols; ++j){
			const T& curr_val = _matrix[j+i*_cols];
			if(curr_val < inner_curr_min){
				inner_curr_min = curr_val;
				minIdxBuff(i, 0) = j; 
			}
		}
	}
	return minIdxBuff;
}
/**
 * Returns a vertical vector matrix containing 
 * the column index of each row-wise max
 * 
 * [[0, 1, 2, 3],	   [[3],
 *  [4, 5, 6, 7],  ->   [3],
 *  [8, 9, 0, 1]]		[1]]
 */
template<typename T>
Matrix<int> Matrix<T>::vMaxIndex(){
	T curr_max = std::numeric_limits<T>::min();
	Matrix<T> maxBuff(_rows, 1, curr_max, _n_threads);
	Matrix<int> maxIdxBuff(_rows, 1, -1, _n_threads);

	#pragma omp parallel for collapse(1) num_threads(_n_threads) if(_threads_enabled)
	for(int i = 0; i < _rows; ++i){
		T& inner_curr_max = maxBuff(i, 0);
		#pragma omp simd
		for(int j = 0; j < _cols; ++j){
			const T& curr_val = _matrix[j+i*_cols];
			if(curr_val > inner_curr_max){ 
				inner_curr_max = curr_val;
				maxIdxBuff(i, 0) = j;
			}
		}
	}
	return maxIdxBuff;
}
/**
 * Min indices of entire matrix
*/
template<typename T>
std::pair<int, int> Matrix<T>::minIndex(){
	Matrix<int> vMinIdx = this->vMinIndex();
	Matrix<T> vMinValues = this->vMin();
	int iMin = vMinValues.hMinIndex()(0, 0);
	int jMin = vMinIdx(iMin, 0);
	return std::pair<int, int>(iMin, jMin);
}
/**
 * Max indices of entire matrix
*/
template<typename T>
std::pair<int, int> Matrix<T>::maxIndex(){
	Matrix<int> vMaxIdx = this->vMaxIndex();
	Matrix<T> vMaxValues = this->vMax();
	int iMax = vMaxValues.hMaxIndex()(0, 0);
	int jMax = vMaxIdx(iMax, 0);
	return std::pair<int, int>(iMax, jMax);
}
/**
 * Returns a horizontal matrix containing column-wise min
 * 
 * [[0, 1, 2, 3],
 *  [4, 5, 6, 7],  ->  [0, 1, 0, 1]
 *  [8, 9, 0, 1]]
 */
template<typename T>
Matrix<T> Matrix<T>::hMin(){
	T curr_min = std::numeric_limits<T>::max();
	Matrix<T> res(1, _cols, curr_min, _n_threads);
	/**
	 * would for(j...){for(i...)} order make it possible to remove atomic write?
	*/
	#pragma omp parallel for collapse(2) num_threads(_n_threads) if(_threads_enabled)
	for(int i = 0; i < _rows; ++i){
		for(int j = 0; j < _cols; ++j){
			T& inner_curr_min = res(0, j);
			const T& curr_val = _matrix[j+i*_cols];
			if(curr_val < inner_curr_min){ 
				#pragma atomic write
				inner_curr_min = curr_val; 
			}
		}
	}
	return res;
}
/**
 * Returns a horizontal matrix containing column-wise max
 * 
 * [[0, 1, 2, 3],
 *  [4, 5, 6, 7],  ->  [8, 9, 6, 7]
 *  [8, 9, 0, 1]]
 */
template<typename T>
Matrix<T> Matrix<T>::hMax(){
	T curr_max = std::numeric_limits<T>::min();
	Matrix<T> res(1, _cols, curr_max, _n_threads);
	/**
	 * would for(j...){for(i...)} order make it possible to remove atomic write?
	*/
	#pragma omp parallel for collapse(1) num_threads(_n_threads) if(_threads_enabled)
	for(int i = 0; i < _rows; ++i){
		#pragma omp simd
		for(int j = 0; j < _cols; ++j){
			T& inner_curr_max = res(0, j);
			const T& curr_val = _matrix[j+i*_cols];
			if(curr_val > inner_curr_max){
				#pragma atomic write 
				inner_curr_max = curr_val; 
			}
		}
	}
	return res;
}
/**
 * Returns a horizontal matrix containing row-wise min
 * 
 * [[0, 1, 2, 3],	   [[0],
 *  [4, 5, 6, 7],  ->   [4],
 *  [8, 9, 0, 1]]		[0]]
 */
template<typename T>
Matrix<T> Matrix<T>::vMin(){
	T curr_min = std::numeric_limits<T>::max();
	Matrix<T> res(_rows, 1, curr_min, _n_threads);
	
	#pragma omp parallel for collapse(1) num_threads(_n_threads) if(_threads_enabled)
	for(int i = 0; i < _rows; ++i){
		T& inner_curr_min = res(i, 0);
		#pragma omp simd
		for(int j = 0; j < _cols; ++j){
			const T& curr_val = _matrix[j+i*_cols];
			if(curr_val < inner_curr_min){ inner_curr_min = curr_val; }
		}
	}
	return res;
}
/**
 * Returns a horizontal matrix containing row-wise max
 * 
 * [[0, 1, 2, 3],	   [[3],
 *  [4, 5, 6, 7],  ->   [7],
 *  [8, 9, 0, 1]]		[9]]
 */
template<typename T>
Matrix<T> Matrix<T>::vMax(){
	T curr_max = std::numeric_limits<T>::min();
	Matrix<T> res(_rows, 1, curr_max, _n_threads);
	
	#pragma omp parallel for collapse(1) num_threads(_n_threads) if(_threads_enabled)
	for(int i = 0; i < _rows; ++i){
		T& inner_curr_max = res(i, 0);
		#pragma omp simd
		for(int j = 0; j < _cols; ++j){
			const T& curr_val = _matrix[j+i*_cols];
			if(curr_val > inner_curr_max){ inner_curr_max = curr_val; }
		}
	}
	return res;
}
/**
 * Min element of entire matrix
*/
template<typename T>
T Matrix<T>::min(){
	return this->vMin().hMin()(0, 0);
}
/**
 * Max element of entire matrix
*/
template<typename T>
T Matrix<T>::max(){
	return this->vMax().hMax()(0, 0);
}

template<typename T>
bool Matrix<T>::isEqual(const Matrix<T>& rhs){
	
	if(!omp_get_cancellation()){
		printf("Enabling thread cancellation...\n");
		putenv("OMP_CANCELLATION=true");
	}
	bool res = true; 
	#pragma omp parallel num_threads(_n_threads) if(_threads_enabled)
	{
		#pragma omp for
		for (int idx = 0; idx < _rows*_cols; ++idx) {
			if(_matrix[idx] != rhs(idx/_cols, idx%_cols)) {
				#pragma omp atomic write
				res = false;
				#pragma omp cancel for
			}
			#pragma omp cancellation point for
		}
	}
	return res;
}

//////////////////////////////////////////////////////////////////////
					/// MATRIX OPERATORS ///
//////////////////////////////////////////////////////////////////////

template<typename T>
T& Matrix<T>::operator()(const int& row, const int& col) {
	return _matrix[col+row*_cols];
}

template<typename T>
const T& Matrix<T>::operator()(const int& row, const int& col) const {
	return _matrix[col+row*_cols];
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& other) {
	int new_rows = other.getRows();
	int new_cols = other.getCols();

	_matrix = std::make_unique<T[]>(new_rows*new_cols);

	#pragma omp parallel for num_threads(_n_threads) if(_threads_enabled)
	for (int index = 0; index < new_rows*new_cols; ++index) {
		_matrix[index] = other(index/new_cols, index%new_cols);
	}
	_rows = new_rows;
	_cols = new_cols;
	return *this;
}

//////////////////////////////////////////////////////////////////////
				/// (MATRIX -- MATRIX) MATH. OPERATIONS ///
//////////////////////////////////////////////////////////////////////

template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) {
	//assert(other.getCols() == m_cols && other.getRows() == m_rows);

	Matrix<T> resultMatrix(_rows, _cols, 0, _n_threads);

	#pragma omp parallel for num_threads(_n_threads) if(_threads_enabled)
	for (int index = 0; index < _rows*_cols; ++index) {
		resultMatrix(index/_cols, index%_cols) = _matrix[index] + other(index / _cols, index%_cols);
	}
	return resultMatrix;
}

template<typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& other) {
	//assert(other.getCols() == m_cols && other.getRows() == m_rows);

	#pragma omp parallel for num_threads(_n_threads) if(_threads_enabled)
	for (int index = 0; index < _rows*_cols; ++index) {
		_matrix[index] += other(index/_cols, index%_cols);
	}
	return *this;
}

template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& other) {
	//assert(other.getCols() == m_cols && other.getRows() == m_rows);

	Matrix<T> resultMatrix(_rows, _cols, 0, _n_threads);

	#pragma omp parallel for num_threads(_n_threads) if(_threads_enabled)
	for (int index = 0; index < _rows*_cols; ++index) {
		resultMatrix(index/_cols, index%_cols) = _matrix[index] - other(index / _cols, index%_cols);
	}
	return resultMatrix;
}

template<typename T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& other) {
	//assert(other.getCols() == m_cols && other.getRows() == m_rows);

	#pragma omp parallel for num_threads(_n_threads) if(_threads_enabled)
	for (int index = 0; index < _rows*_cols; ++index) {
		_matrix[index] -= other(index/_cols, index%_cols);
	}
	return *this;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) {
	//assert(other.getCols() == m_cols && other.getRows() == m_rows);

	Matrix<T> resultMatrix(_rows, _cols, 0, _n_threads);

	#pragma omp parallel for num_threads(_n_threads) if(_threads_enabled)
	for (int index = 0; index < _rows*_cols; ++index) {
		resultMatrix(index / _cols, index%_cols) = _matrix[index] * other(index / _cols, index%_cols);
	}
	return resultMatrix;
}

template<typename T>
Matrix<T>& Matrix<T>::operator*=(const Matrix<T>& other) {
	//assert(other.getCols() == m_cols && other.getRows() == m_rows);

	#pragma omp parallel for num_threads(_n_threads) if(_threads_enabled)
	for (int index = 0; index < _rows*_cols; ++index) {
		_matrix[index] *= other(index / _cols, index%_cols);
	}
	return *this;
}

//////////////////////////////////////////////////////////////////////
			/// (MATRIX -- SCALAR) MATH. OPERATIONS (RHS) ///
//////////////////////////////////////////////////////////////////////

template<typename T>
Matrix<T> Matrix<T>::operator+(T scalar) {
	Matrix<T> resultMatrix(_rows, _cols, 0, _n_threads);

	#pragma omp parallel for num_threads(_n_threads) if(_threads_enabled)
	for (int index = 0; index < _rows*_cols; ++index) {
		resultMatrix(index / _cols, index%_cols) = _matrix[index] + scalar;
	}
	return resultMatrix;
}

template<typename T>
Matrix<T> Matrix<T>::operator-(T scalar) {
	Matrix<T> resultMatrix(_rows, _cols, 0, _n_threads);

	#pragma omp parallel for num_threads(_n_threads) if(_threads_enabled)
	for (int index = 0; index < _rows*_cols; ++index) {
		resultMatrix(index / _cols, index%_cols) = _matrix[index] - scalar;
	}
	return resultMatrix;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(T scalar) {
	Matrix<T> resultMatrix(_rows, _cols, 0, _n_threads);

	#pragma omp parallel for num_threads(_n_threads) if(_threads_enabled)
	for (int index = 0; index < _rows*_cols; ++index) {
		resultMatrix(index / _cols, index%_cols) = _matrix[index] * scalar;
	}
	return resultMatrix;
}

template<typename T>
Matrix<T> Matrix<T>::operator/(T scalar) {
	assert(scalar != 0);
	Matrix<T> resultMatrix(_rows, _cols, 0, _n_threads);

	#pragma omp parallel for num_threads(_n_threads) if(_threads_enabled)
	for (int index = 0; index < _rows*_cols; ++index) {
		resultMatrix(index / _cols, index%_cols) = _matrix[index] / scalar;
	}
	return resultMatrix;
}

template<typename T>
Matrix<T>& Matrix<T>::operator+=(T scalar) {
	#pragma omp parallel for num_threads(_n_threads) if(_threads_enabled)
	for (int index = 0; index < _rows*_cols; ++index) {
		_matrix[index] = _matrix[index] + scalar;
	}
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator*=(T scalar) {
	#pragma omp parallel for num_threads(_n_threads) if(_threads_enabled)
	for (int index = 0; index < _rows*_cols; ++index) {
		_matrix[index] = _matrix[index] * scalar;
	}
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator-=(T scalar) {
	#pragma omp parallel for num_threads(_n_threads) if(_threads_enabled)
	for (int index = 0; index < _rows*_cols; ++index) {
		_matrix[index] = _matrix[index] - scalar;
	}
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator/=(T scalar) {
	assert(scalar != 0);
	#pragma omp parallel for num_threads(_n_threads) if(_threads_enabled)
	for (int index = 0; index < _rows*_cols; ++index) {
		_matrix[index] = _matrix[index] / scalar;
	}
	return *this;
}

//////////////////////////////////////////////////////////////////////
					/// MATRIX VISUALIZATION ///
//////////////////////////////////////////////////////////////////////

template<typename T>
void Matrix<T>::print(std::ostream& STREAM) const {

	for (size_t i = 0; i < _rows; ++i) {
		for (size_t j = 0; j < _cols; ++j) {
			STREAM << _matrix[j + _cols * i];
			if(j < _cols-1) STREAM << ", ";
		}
		if(i < _rows-1) STREAM << std::endl;
	}
	//STREAM << std::endl;
}
