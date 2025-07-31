// #include <omp.h>
#include "walltime.h"
#include <iostream>
#include <math.h>
#include <stdio.h>

#define NUM_ITERATIONS 100
#define N_DEF 100000
#define EPSILON 0.1

using namespace std;

int main(int argc, char *argv[])
{
    int N = N_DEF;
    double time_red, time_start;
    double *a, *b;
  
    if (argc == 2)
        N = stoi(argv[1]);
  
    a = new double[N];
    b = new double[N];
  
    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i / 10.0;
    }
  
    volatile long double alpha_parallel = 0;
  
    // Parallel version using reduction
    time_start = wall_time();
    for (int iterations = 0; iterations < NUM_ITERATIONS; iterations++)
    {
        alpha_parallel = 0.0;
        #pragma omp parallel for default(none) reduction(+ : alpha_parallel) shared(a, b, N)
        for (int i = 0; i < N; i++)
        {
            alpha_parallel += a[i] * b[i];
        }
    }
    time_red = wall_time() - time_start;
    cout << time_red; // Output just the execution time

    delete[] a;
    delete[] b;

    return 0;
}
