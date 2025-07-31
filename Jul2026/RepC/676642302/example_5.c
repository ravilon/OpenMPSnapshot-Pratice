#include <stdio.h>
#include <math.h>
#include <omp.h>

double f(double a);

int main(int argc, char **argv) {
    int n, i;
    double PI25DT = 3.141592653589793238462643;
    double pi, h, sum, x;

    n = 32; // Initializes the number of intervals as `n`
    h = 1. / (double)n;
    sum = 0.;

    // The pi variable is a reduction variable by sum
    #pragma omp parallel for reduction(+:pi) private(x, i)
    for (i = 1; i <= n; i++) {
        x = h * ((double)i - 0.5);
        pi += f(x);
    }
    
    pi = h * pi;
    printf("pi is approximately %.16f, error is %.16f\n", pi, fabs(pi - PI25DT));
    return 0;
}

double f(double a) {
    return (4. / (1. + a * a));
}
