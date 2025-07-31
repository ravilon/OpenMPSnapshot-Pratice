#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <omp.h>

#include "a2-helpers.hpp"

using namespace std;

int main(int argc, char **argv)
{
    // default program parameters
    int N = 12;
    int M = 12;
    int max_iterations = 1000;
    double epsilon = 1.0e-3;
    bool verify = true, print_config = false;

    process_input(argc, argv, N, M, max_iterations, epsilon, verify, print_config);

    if ( print_config )
        std::cout << "Configuration: m: " << M << ", n: " << N << ", max-iterations: " << max_iterations << ", epsilon: " << epsilon << std::endl;

    #if defined(_OPENMP)
        auto time_1 = omp_get_wtime();
    #else
        auto time_1 = chrono::high_resolution_clock::now();
    #endif

    // DO NOT MODIFY code above this comment!

    int i, j;

    // Matrix Declaration & Allocation
    Mat U(M, N);
    Mat W(M, N);

    // Init & Boundary
    #pragma omp parallel for default(none) shared(U, W) private(i, j) firstprivate(M, N)
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            W[i][j] = U[i][j] = 0.0;
        }

        W[i][0] = U[i][0] = 0.05;  // left side
        W[i][N - 1] = U[i][N - 1] = 0.1;  // right side
    }
    #pragma omp parallel for default(none) shared(U, W) private(j) firstprivate(M, N)
    for (j = 0; j < N; ++j)
    {
        W[0][j] = U[0][j] = 0.02;  // top 
        W[M - 1][j] = U[M - 1][j] = 0.2;  // bottom 
    }
    // End init
    int iteration_count = 0;
    double diffnorm;
    do
    {
        iteration_count++;
        diffnorm = 0.0;

        // Compute new values (but not on boundary)
        #pragma omp parallel for default(none) shared(U, W) private(i, j) reduction(+:diffnorm) firstprivate(M, N)
        for (i = 1; i < M - 1; ++i)
        {
            for (j = 1; j < N - 1; ++j)
            {
                W[i][j] = (U[i][j + 1] + U[i][j - 1] + U[i + 1][j] + U[i - 1][j]) * 0.25;
                diffnorm += (W[i][j] - U[i][j]) * (W[i][j] - U[i][j]);
            }
        }

        // Copy interior points from matrix W to U
        #pragma omp parallel for default(none) shared(U, W) private(i, j) firstprivate(M, N)
        for (i = 1; i < M - 1; ++i)
        {
            for (j = 1; j < N - 1; ++j)
            {
                U[i][j] = W[i][j];
            }
        }
        diffnorm = sqrt(diffnorm); // all processes need to know when to stop
        } while (epsilon <= diffnorm && iteration_count < max_iterations);

    // DO NOT MODIFY code below this comment!

    // Print time measurements
    std::cout << "Elapsed time: ";
    #if defined(_OPENMP)
        auto time_2 = omp_get_wtime();
        std::cout << std::fixed << std::setprecision(4) << (time_2 - time_1);
    #else
        auto time_2 = chrono::high_resolution_clock::now();
        std::cout << std::fixed << std::setprecision(4) << chrono::duration<double>(time_2 - time_1).count();
    #endif
    std::cout << " seconds, iterations: " << iteration_count << std::endl;

    // verification
    if ( verify ) {
        Mat U_sequential(M, N); // init another matrix for the verification

        int iteration_count_seq = 0;
        heat2d_sequential(U_sequential, max_iterations, epsilon, iteration_count_seq);

        // Elementwise comparison of matrix U and matrix U_sequential
        cout << "Verification: " << ( U.compare(U_sequential) && iteration_count == iteration_count_seq ? "OK" : "NOT OK") << std::endl;
    }

    // U.save_to_disk("heat2d.txt"); // not needed

    return 0;
}
