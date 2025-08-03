// This program evaluates PI with CPU with OpenMP using the Monte Carlo method
// g++ -fopenmp -pedantic -pipe -O3 -march=native pi_omp.cpp -o pi_omp

#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <math.h>
#include <omp.h>

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

#define THREADS 12

int main(int argc, char **argv)
{
    // Choice the random seed
    srand(time(NULL));

    // Total points inside circle
    int sum = 0;

    // Iterations number
    int N = 256 * 256 * atoi(argv[1]);

    // Random coordinates from the range (0,1)
    double x, y;

    // Execution time - start
    auto start = high_resolution_clock::now();

// Launch sum calculation
#pragma omp parallel for private(x, y) reduction(+ : sum) num_threads(THREADS)
    for (int i = 0; i < N; i++)
    {
        x = (double)rand() / RAND_MAX;
        y = (double)rand() / RAND_MAX;

        if (x * x + y * y <= 1.0)
            sum++;
    }

    // Execution time - stop
    auto stop = high_resolution_clock::now();

    // Evaluate PI
    double pi = sum * 4.0 / N;

    // Evaluate relative error
    double err = abs(pi - acos(-1)) / pi * 100;

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = stop - start;

    printf("\nCompleted successfully!\n");
    printf("CPU PI = %f\n", pi);
    printf("CPU relative error = %f pct\n", err);
    printf("calculatePi() execution time on the CPU: %f ms\n", ms_double.count());

    return 0;
}