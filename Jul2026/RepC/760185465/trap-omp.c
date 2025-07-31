#include <iostream>
#include <cmath>
#include <cstdlib>
#include <omp.h>
using namespace std;

const double pi = 3.141592653589793238462643383079;

int main(int argc, char** argv) {
    double a = 0.0, b = pi;
    int n = 1048576;
    double h = (b - a) / n;
    double integral = 0.0;
    int threadct = 1;

    if (argc > 1)
        threadct = atoi(argv[1]);

    double f(double x);

#ifdef _OPENMP
    cout << "OMP defined, threadct = " << threadct << endl;
#else
    cout << "OMP not defined" << endl;
#endif

    double start = omp_get_wtime(); // Starting the time measurement

    integral = (f(a) + f(b)) / 2.0;

#pragma omp parallel for num_threads(threadct) reduction(+:integral)
    for (int i = 1; i < n; i++) {
        integral += f(a + i * h);
    }

    integral *= h;

    double end = omp_get_wtime(); // Measuring the elapsed time
    double elapsed_time = end - start; // Time calculation

    cout << "With n = " << n << " trapezoids, our estimate of the integral from " 
         << a << " to " << b << " is " << integral << ".\n";
    cout << "Elapsed time: " << elapsed_time << " seconds.\n";
}

double f(double x) {
    return sin(x);
}
