#include <iostream>
#include <vector>
#include "BaseMatrix.cpp"

class StrassenMatrix : public BaseMatrix {
public:
StrassenMatrix() = default;

StrassenMatrix(int rows, int cols) : BaseMatrix(rows, cols) {}

public:
//      
void splitMatrix(int rowStart, int colStart, int numRows, int numCols, StrassenMatrix& a11, StrassenMatrix& a12, StrassenMatrix& a21, StrassenMatrix& a22) const {
for (int i = 0; i < numRows; ++i) {
for (int j = 0; j < numCols; ++j) {
if (i < numRows / 2 && j < numCols / 2) {
a11(i, j) = data[rowStart + i][colStart + j];
}
else if (i < numRows / 2) {
a12(i, j - numCols / 2) = data[rowStart + i][colStart + j];
}
else if (j < numCols / 2) {
a21(i - numRows / 2, j) = data[rowStart + i][colStart + j];
}
else {
a22(i - numRows / 2, j - numCols / 2) = data[rowStart + i][colStart + j];
}
}
}
}

void splitMatrixParallel(int rowStart, int colStart, int numRows, int numCols, StrassenMatrix& a11, StrassenMatrix& a12, StrassenMatrix& a21, StrassenMatrix& a22) const {
#pragma omp parallel for collapse(2) schedule(dynamic) shared(rowStart, colStart, numRows, numCols, a11, a12, a21, a22)
for (int i = 0; i < numRows; ++i) {
for (int j = 0; j < numCols; ++j) {
if (i < numRows / 2 && j < numCols / 2) {
a11(i, j) = data[rowStart + i][colStart + j];
}
else if (i < numRows / 2) {
a12(i, j - numCols / 2) = data[rowStart + i][colStart + j];
}
else if (j < numCols / 2) {
a21(i - numRows / 2, j) = data[rowStart + i][colStart + j];
}
else {
a22(i - numRows / 2, j - numCols / 2) = data[rowStart + i][colStart + j];
}
}
}
}


//      
void collectMatrix(int rowStart, int colStart, const StrassenMatrix& c11, const StrassenMatrix& c12, const StrassenMatrix& c21, const StrassenMatrix& c22) {
int numRows = c11.rows();
int numCols = c11.cols();

for (int i = 0; i < numRows; ++i) {
for (int j = 0; j < numCols; ++j) {
data[rowStart + i][colStart + j] = c11(i, j);
}
}

for (int i = 0; i < numRows; ++i) {
for (int j = 0; j < c12.cols(); ++j) {
data[rowStart + i][colStart + numCols + j] = c12(i, j);
}
}

for (int i = 0; i < c21.rows(); ++i) {
for (int j = 0; j < numCols; ++j) {
data[rowStart + numRows + i][colStart + j] = c21(i, j);
}
}

for (int i = 0; i < c22.rows(); ++i) {
for (int j = 0; j < c22.cols(); ++j) {
data[rowStart + numRows + i][colStart + numCols + j] = c22(i, j);
}
}
}

void collectMatrixParallel(int rowStart, int colStart, const StrassenMatrix& c11, const StrassenMatrix& c12, const StrassenMatrix& c21, const StrassenMatrix& c22) {
int numRows = c11.rows();
int numCols = c11.cols();

#pragma omp parallel for schedule(dynamic) shared(rowStart, colStart, numRows, numCols, c11, c12, c21, c22)
for (int i = 0; i < numRows; ++i) {
// Collect C11
for (int j = 0; j < numCols; ++j) {
data[rowStart + i][colStart + j] = c11(i, j);
}

// Collect C12
for (int j = 0; j < c12.cols(); ++j) {
data[rowStart + i][colStart + numCols + j] = c12(i, j);
}
}

#pragma omp parallel for schedule(dynamic) shared(rowStart, colStart, numRows, c21, numCols)
for (int i = 0; i < c21.rows(); ++i) {
// Collect C21
for (int j = 0; j < numCols; ++j) {
data[rowStart + numRows + i][colStart + j] = c21(i, j);
}
}

#pragma omp parallel for schedule(dynamic) shared(rowStart, colStart, numRows, numCols, c22)
for (int i = 0; i < c22.rows(); ++i) {
// Collect C22
for (int j = 0; j < c22.cols(); ++j) {
data[rowStart + numRows + i][colStart + numCols + j] = c22(i, j);
}
}
}


//    
StrassenMatrix summation(const StrassenMatrix& other) const {
int numRows = rows();
int numCols = cols();
StrassenMatrix result(numRows, numCols);

for (int i = 0; i < numRows; ++i) {
for (int j = 0; j < numCols; ++j) {
result(i, j) = data[i][j] + other(i, j);
}
}

return result;
}

//    
StrassenMatrix subtraction(const StrassenMatrix& other) const {
int numRows = rows();
int numCols = cols();
StrassenMatrix result(numRows, numCols);

for (int i = 0; i < numRows; ++i) {
for (int j = 0; j < numCols; ++j) {
result(i, j) = data[i][j] - other(i, j);
}
}

return result;
}

//     
int rows() const {
return data.size();
}

int cols() const {
return data[0].size();
}

//      
int& operator()(int i, int j) {
return data[i][j];
}

//       
const int& operator()(int i, int j) const {
return data[i][j];
}

//     ( )
StrassenMatrix multiplySequential(const StrassenMatrix& other) const {
int M = rows();
int N = cols();
int K = other.cols();

StrassenMatrix result(M, K);

for (int i = 0; i < M; ++i) {
for (int j = 0; j < K; ++j) {
for (int k = 0; k < N; ++k) {
result(i, j) += data[i][k] * other(k, j);
}
}
}

return result;
}

StrassenMatrix operator+(const StrassenMatrix& other) const {
int numRows = rows();
int numCols = cols();
StrassenMatrix result(numRows, numCols);

for (int i = 0; i < numRows; ++i) {
for (int j = 0; j < numCols; ++j) {
result(i, j) = data[i][j] + other(i, j);
}
}

return result;
}

//   
StrassenMatrix operator-(const StrassenMatrix& other) const {
int numRows = rows();
int numCols = cols();
StrassenMatrix result(numRows, numCols);

for (int i = 0; i < numRows; ++i) {
for (int j = 0; j < numCols; ++j) {
result(i, j) = data[i][j] - other(i, j);
}
}

return result;
}

//     
StrassenMatrix operator+(int scalar) const {
int numRows = rows();
int numCols = cols();
StrassenMatrix result(numRows, numCols);

for (int i = 0; i < numRows; ++i) {
for (int j = 0; j < numCols; ++j) {
result(i, j) = data[i][j] + scalar;
}
}

return result;
}

//     
StrassenMatrix operator-(int scalar) const {
int numRows = rows();
int numCols = cols();
StrassenMatrix result(numRows, numCols);

for (int i = 0; i < numRows; ++i) {
for (int j = 0; j < numCols; ++j) {
result(i, j) = data[i][j] - scalar;
}
}

return result;
}

StrassenMatrix parallelAdd(const StrassenMatrix& other) const {
int numRows = rows();
int numCols = cols();
StrassenMatrix result(numRows, numCols);

#pragma omp parallel for collapse(2) schedule(dynamic) shared(numRows, numCols, result, other)
for (int i = 0; i < numRows; ++i) {
for (int j = 0; j < numCols; ++j) {
result(i, j) = (*this)(i, j) + other(i, j);
}
}

return result;
}

StrassenMatrix parallelSubtract(const StrassenMatrix& other) const {
int numRows = rows();
int numCols = cols();
StrassenMatrix result(numRows, numCols);

#pragma omp parallel for collapse(2) schedule(dynamic) shared(numRows, numCols, result, other)
for (int i = 0; i < numRows; ++i) {
for (int j = 0; j < numCols; ++j) {
result(i, j) = (*this)(i, j) - other(i, j);
}
}

return result;
}

//     (  )
StrassenMatrix multiplySequentialStrassen(const StrassenMatrix& other) const {
int M = rows();
int N = cols();
int K = other.cols();

if (M <= 64 || N <= 64 || K <= 64) {
//    ,   
return multiplySequential(other);
}

StrassenMatrix result(M, K);

int n = M / 2;

//    
StrassenMatrix A11(n, n), A12(n, N - n), A21(M - n, n), A22(M - n, N - n);
StrassenMatrix B11(n, K - n), B12(n, n), B21(K - n, K - n), B22(K - n, n);

splitMatrix(0, 0, n, n, A11, A12, A21, A22);
other.splitMatrix(0, 0, n, K - n, B11, B12, B21, B22);

//  
StrassenMatrix P1 = A11.multiplySequentialStrassen(B12 - B22);
StrassenMatrix P2 = (A11 + A12).multiplySequentialStrassen(B22);
StrassenMatrix P3 = (A21 + A22).multiplySequentialStrassen(B11);
StrassenMatrix P4 = A22.multiplySequentialStrassen(B21 - B11);
StrassenMatrix P5 = (A11 + A22).multiplySequentialStrassen(B11 + B22);
StrassenMatrix P6 = (A12 - A22).multiplySequentialStrassen(B21 + B22);
StrassenMatrix P7 = (A11 - A21).multiplySequentialStrassen(B11 + B12);

//   
StrassenMatrix C11 = P5 + P4 - P2 + P6;
StrassenMatrix C12 = P1 + P2;
StrassenMatrix C21 = P3 + P4;
StrassenMatrix C22 = P5 + P1 - P3 - P7;

//     
result.collectMatrix(0, 0, C11, C12, C21, C22);

return result;
}

//     (  )
StrassenMatrix multiplyParallelStrassen(const StrassenMatrix& other) const {
int M = rows();
int N = cols();
int K = other.cols();

if (M <= 64 || N <= 64 || K <= 64) {
//    ,   
return multiplySequential(other);
}

StrassenMatrix result(M, K);

int n = M / 2;

//    
StrassenMatrix A11(n, n), A12(n, N - n), A21(M - n, n), A22(M - n, N - n);
StrassenMatrix B11(n, K - n), B12(n, n), B21(K - n, K - n), B22(K - n, n);

splitMatrixParallel(0, 0, n, n, A11, A12, A21, A22);
other.splitMatrixParallel(0, 0, n, K - n, B11, B12, B21, B22);

//  

StrassenMatrix P1 = A11.multiplySequentialStrassen(B12.parallelSubtract(B22));
StrassenMatrix P2 = A11.parallelAdd(A12).multiplySequentialStrassen(B22);
StrassenMatrix P3 = A21.parallelAdd(A22).multiplySequentialStrassen(B11);
StrassenMatrix P4 = A22.multiplySequentialStrassen(B21.parallelSubtract(B11));
StrassenMatrix P5 = A11.parallelAdd(A22).multiplySequentialStrassen(B11.parallelAdd(B11));
StrassenMatrix P6 = A12.parallelSubtract(A22).multiplySequentialStrassen(B21.parallelAdd(B22));
StrassenMatrix P7 = A11.parallelSubtract(A21).multiplySequentialStrassen(B11.parallelAdd(B12));

//   
StrassenMatrix C11 = P5.parallelAdd(P4).parallelSubtract(P2).parallelAdd(P6);
StrassenMatrix C12 = P1.parallelAdd(P2);
StrassenMatrix C21 = P3.parallelAdd(P4);
StrassenMatrix C22 = P5.parallelAdd(P1).parallelSubtract(P3).parallelSubtract(P7);

//     
result.collectMatrixParallel(0, 0, C11, C12, C21, C22);

return result;
}
};
