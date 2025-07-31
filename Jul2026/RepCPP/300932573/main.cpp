#include <iostream>
#include <immintrin.h>
#include <time.h>
#include <vector>
#include <ctime>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <thread>

using namespace std;

/*
*Given arrays of widths and heights, calculate total area linearly
*/
double linear_operation(double *w, double *h, int n)
{
    double total = 0;
    for (int i = 0; i < n; i++)
    {
        total += w[i] * h[i];
    }
    return total;
}

/*
*Given arrays of widths and heights, calculate total area using OpenMP
*/
double openmp_operation(double *w, double *h, int n)
{
    omp_set_num_threads((int)(std::thread::hardware_concurrency()));
    double total = 0;

    #pragma omp parallel for reduction(+:total)
    for (int i = 0; i < n; i++)
    {
        total += w[i] * h[i];
    }
    return total;
}

/*Start of main program*/
int main()
{
    //Generate some random values for widths and heights
    int n = 100000; //numbers of widths and heights

    double w[n];
    double h[n];

    srand(time(0)); //Set seed for rand

    for (int i = 0; i < n - (n % 8); i++)
    {
        w[i] = (((double)rand()) / RAND_MAX) * 100 - 50;
        h[i] = (((double)rand()) / RAND_MAX) * 100 - 50;
    }

    //Start testing
    int iter = 100000; //number of iteration to run each functions

    //Normal operations
    double linear_result = 0;

    chrono::steady_clock::time_point start = chrono::steady_clock::now();

    for (int i = 0; i < iter; i++)
    {
        linear_result = linear_operation(w, h, n);
    }

    chrono::steady_clock::time_point finish = chrono::steady_clock::now();
    double linear_time = chrono::duration_cast<std::chrono::duration<double>>(finish - start).count() / iter;

    //OpenMP operations
    double openmp_result = 0;
    start = chrono::steady_clock::now();
    for (int i = 0; i < iter; i++)
    {
        openmp_result = openmp_operation(w, h, n);
    }
    finish = chrono::steady_clock::now();
    double openmp_time = chrono::duration_cast<std::chrono::duration<double>>(finish - start).count() / iter;

    //Print results

    cout << "Number of iteration : " << iter << endl;
    cout << "Number of thread : " << (int)std::thread::hardware_concurrency() << endl;

    cout << "Linear operations result: " << linear_result << endl;
    cout << "OpenMP operations result: " << openmp_result << endl;

    cout << "Averaged time elpased is " << linear_time << " seconds for linear operations" << endl;
    cout << "Averaged Time elpased is " << openmp_time << " seconds for OpenMP operations" << endl;

    cout << "Speedup: " << linear_time / openmp_time << " x " << endl;

    return 0;
}
