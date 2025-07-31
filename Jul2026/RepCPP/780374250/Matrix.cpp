#include <iostream>
#include "BaseMatrix.cpp"
#include <vector>

class Matrix : public BaseMatrix {
public:
    Matrix() = default;

    Matrix(int rows, int cols) : BaseMatrix(rows, cols) {}

    //   
    Matrix multiplySequential(const Matrix& other) const {
        int M = data.size();
        int N = data[0].size();
        int K = other.data[0].size();

        Matrix result(M, K);

        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < K; ++j) {
                for (int k = 0; k < N; ++k) {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }

        return result;
    }

    //      for static
    Matrix multiplyParallelStatic(const Matrix& other) const {
        int M = data.size();
        int N = data[0].size();
        int K = other.data[0].size();

        Matrix result(M, K);

        int i, j, k;
#pragma omp parallel for collapse(2) schedule(static) private(i, j, k)
        for (i = 0; i < M; ++i) {
            for (j = 0; j < K; ++j) {
                for (k = 0; k < N; ++k) {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }

        return result;
    }

    //      for dynamic
    Matrix multiplyParallelDynamic(const Matrix& other) const {
        int M = data.size();
        int N = data[0].size();
        int K = other.data[0].size();

        Matrix result(M, K);

        int i, j, k;
#pragma omp parallel for collapse(2) schedule(dynamic) private(i, j, k)
        for (i = 0; i < M; ++i) {
            for (j = 0; j < K; ++j) {
                for (k = 0; k < N; ++k) {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }

        return result;
    }

    //      for guided
    Matrix multiplyParallelGuided(const Matrix& other) const {
        int M = data.size();
        int N = data[0].size();
        int K = other.data[0].size();

        Matrix result(M, K);

        int i, j, k;
#pragma omp parallel for collapse(2) schedule(guided) private(i, j, k)
        for (i = 0; i < M; ++i) {
            for (j = 0; j < K; ++j) {
                for (k = 0; k < N; ++k) {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }

        return result;
    }

    //      sections
    Matrix multiplyParallelSections(const Matrix& other) const {
        int M = data.size();
        int N = data[0].size();
        int K = other.data[0].size();

        Matrix result(M, K);

        int i, j, k;
#pragma omp parallel sections private(i, j, k)
        {
#pragma omp section 
            {
                for (int i = 0; i < M / 2; ++i) {
                    for (int j = 0; j < K; ++j) {
                        for (int k = 0; k < N; ++k) {
                            result.data[i][j] += data[i][k] * other.data[k][j];
                        }
                    }
                }
            }

#pragma omp section 
            {
                for (int i = M / 2; i < M; ++i) {
                    for (int j = 0; j < K; ++j) {
                        for (int k = 0; k < N; ++k) {
                            result.data[i][j] += data[i][k] * other.data[k][j];
                        }
                    }
                }
            }
        }

        return result;
    }
};