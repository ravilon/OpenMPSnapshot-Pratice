#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void trap(double a, double b, int n, double* globbal_result );

int main(int argc, char *argv []) {
    double global_result = 0.0;
    double a, b;
    int n;

    int thread_count = strtol(argv[1], NULL, 10);

    printf("Enter a, b, n\n");
    scanf("%lf %lf %d", &a, &b, &n);
    
    #pragma omp parallel num_threads(thread_count) 
    trap(a, b, n, &global_result);

    printf("With n = %d trapezoids, our estimate\n", n);
    printf("of the integral from %f to %f = %.14e\n", a, b, global_result);

}

double func(double x) {
    return x * x;
}


void trap(double a, double b, int n, double* global_result ) {
    int my_rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();

    double h = (b - a) / n;
    int local_n = n / thread_count;

    double local_a = a +  my_rank * local_n * h;
    double local_b = local_a + local_n * h;

    double my_result = (func(local_a) + func(local_b)) / 2.0;
    for (int i = 1; i <= local_n - 1; i++) {
        double x_i = local_a + h * i;
        my_result += func(x_i);
    }
    my_result = my_result * h;

    #pragma omp critical
    *global_result += my_result;
}

// gcc -o omp_trapezoidal_area omp_trapezoidal_area.c -fopenmp
