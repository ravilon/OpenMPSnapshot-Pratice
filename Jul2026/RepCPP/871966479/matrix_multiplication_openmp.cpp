#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

// Function to generate random matrices
void generateMatrix(vector<vector<int>> &matrix, int size) {
for (int i = 0; i < size; i++) {
for (int j = 0; j < size; j++) {
matrix[i][j] = rand() % 100;
}
}
}

// Sequential matrix multiplication
void sequentialMatrixMultiplication(const vector<vector<int>> &A, const vector<vector<int>> &B, vector<vector<int>> &C, int size) {
for (int i = 0; i < size; i++) {
for (int j = 0; j < size; j++) {
C[i][j] = 0;
for (int k = 0; k < size; k++) {
C[i][j] += A[i][k] * B[k][j];
}
}
}
}

// Parallelized matrix multiplication using OpenMP
void parallelMatrixMultiplication(const vector<vector<int>> &A, const vector<vector<int>> &B, vector<vector<int>> &C, int size, int num_threads) {
#pragma omp parallel for num_threads(num_threads)
for (int i = 0; i < size; i++) {
for (int j = 0; j < size; j++) {
C[i][j] = 0;
for (int k = 0; k < size; k++) {
C[i][j] += A[i][k] * B[k][j];
}
}
}
}

// Sequential convolution method
void sequentialConvolution(const vector<vector<int>> &A, const vector<vector<int>> &B, vector<vector<int>> &C, int size) {
int kernelSize = 3; // Using a 3x3 kernel
for (int i = 0; i <= size - kernelSize; i++) {
for (int j = 0; j <= size - kernelSize; j++) {
C[i][j] = 0;
for (int ki = 0; ki < kernelSize; ki++) {
for (int kj = 0; kj < kernelSize; kj++) {
C[i][j] += A[i + ki][j + kj] * B[ki][kj];
}
}
}
}
}

// Parallelized convolution method
void parallelConvolution(const vector<vector<int>> &A, const vector<vector<int>> &B, vector<vector<int>> &C, int size, int num_threads) {
int kernelSize = 3; // Using a 3x3 kernel
#pragma omp parallel for num_threads(num_threads)
for (int i = 0; i <= size - kernelSize; i++) {
for (int j = 0; j <= size - kernelSize; j++) {
C[i][j] = 0;
for (int ki = 0; ki < kernelSize; ki++) {
for (int kj = 0; kj < kernelSize; kj++) {
C[i][j] += A[i + ki][j + kj] * B[ki][kj];
}
}
}
}
}

// Function to print the time taken
void printTimeTaken(const string &method, steady_clock::time_point start, steady_clock::time_point end) {
auto duration = duration_cast<chrono::duration<double, milli>>(end - start).count();
printf("%s time taken: %.2f ms\n", method.c_str(), duration);
}

int main() {
srand(static_cast<unsigned int>(time(0)));

vector<int> sizes = {10, 50, 100, 500}; // Matrix sizes
for (int size : sizes) {
// Initialize matrices
vector<vector<int>> A(size, vector<int>(size));
vector<vector<int>> B(size, vector<int>(size));
vector<vector<int>> C(size, vector<int>(size));
vector<vector<int>> C_conv(size, vector<int>(size)); // For convolution method

// Generate random matrices A and B
generateMatrix(A, size);
generateMatrix(B, size);

cout << "Matrix Size: " << size << "x" << size << endl;

// Sequential Matrix Multiplication
auto start = steady_clock::now();
sequentialMatrixMultiplication(A, B, C, size);
auto end = steady_clock::now();
printTimeTaken("Sequential", start, end);

// Parallel Matrix Multiplication with 1, 2, 4, and 8 threads
for (int num_threads : {1, 2, 4, 8}) {
start = steady_clock::now();
parallelMatrixMultiplication(A, B, C, size, num_threads);
end = steady_clock::now();
printf("Parallel (%d threads) version time taken: %.2f ms\n", num_threads, duration_cast<duration<double, milli>>(end - start).count());
}
cout << "--------------------------" << endl;

// Sequential Convolution
start = steady_clock::now();
sequentialConvolution(A, B, C_conv, size);
end = steady_clock::now();
printTimeTaken("Sequential Convolution", start, end);

// Parallel Convolution with 1, 2, 4, and 8 threads
for (int num_threads : {1, 2, 4, 8}) {
start = steady_clock::now();
parallelConvolution(A, B, C_conv, size, num_threads);
end = steady_clock::now();
printf("Parallel Convolution (%d threads) version time taken: %.2f ms\n", num_threads, duration_cast<duration<double, milli>>(end - start).count());
}
cout << "--------------------------" << endl;
}

return 0;
}
