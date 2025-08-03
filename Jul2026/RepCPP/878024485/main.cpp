#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <iomanip>

int main() {
    const int N = 1000;
    const int M = 1000;
    const int K = 500;

    // Create 3D arrays for computation
    std::vector<std::vector<std::vector<double>>> array1(N, 
        std::vector<std::vector<double>>(M, std::vector<double>(K, 0)));
    std::vector<std::vector<std::vector<double>>> array2(N, 
        std::vector<std::vector<double>>(M, std::vector<double>(K, 0)));
    std::vector<std::vector<std::vector<double>>> result(N, 
        std::vector<std::vector<double>>(M, std::vector<double>(K, 0)));

    // Initialize array1 with some values
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            for (int k = 0; k < K; k++) {
                array1[i][j][k] = sin(i) + cos(j) + tan(k);
            }
        }
    }

    // Initialize array2 with a linear sequence
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            for (int k = 0; k < K; k++) {
                array2[i][j][k] = sqrt(i + 1) * log(j + 1) + exp(k % 10);
            }
        }
    }

    // Perform a complex operation on array1 and array2
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            for (int k = 0; k < K; k++) {
                result[i][j][k] = pow(array1[i][j][k], 2) + array2[i][j][k] * sin(k);
            }
        }
    }

    // Calculate a global sum of the result array
    double globalSum = 0.0;
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            for (int k = 0; k < K; k++) {
                globalSum += result[i][j][k];
            }
        }
    }

    // Output a sample value and global sum to confirm calculations
    std::cout << "Computation complete." << std::endl;
    std::cout << "Sample value: " << result[N/2][M/2][K/2] << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "Global sum: " << globalSum << std::endl;

    return 0;
}
