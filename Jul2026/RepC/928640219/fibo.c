#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int fib_recursivo(int n)
{
    int r1, r2;
    if (n < 2)
        return n;

    #pragma omp task shared(r1) firstprivate(n)
    r1 = fib_recursivo(n - 1);

    #pragma omp task shared(r2) firstprivate(n)
    r2 = fib_recursivo(n - 2);

    #pragma omp taskwait
    return r1 + r2;
}

int main(int argc, char **argv)
{
    int r;
    int n = 5;
    double start, end;

    start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        r = fib_recursivo(n);
    }
    end = omp_get_wtime();
    printf("Recursivo\n");
    printf("- %d -\n", r);
    printf("Tempo: %4.2f\n", end - start);

    return 0;
}
