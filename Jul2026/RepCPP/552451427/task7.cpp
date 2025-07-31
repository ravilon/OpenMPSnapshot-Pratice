#include <stdio.h>
#include <omp.h>

int main() {
    const int SIZE = 12;
    int a[12];
    int b[12];
    int c[12];

    omp_set_num_threads(3);
    printf("\nStatic area with total amount of threads: %d\n", omp_get_max_threads());
    #pragma omp parallel for schedule(static, 5)
    for (int i = 0; i < SIZE; i++) 
    {
        printf("  thread №: %d, total amount of threads: %d\n", i, omp_get_thread_num(), omp_get_num_threads());
        a[i] = i * i;
        b[i] = i * i * i;
        printf("    a[%d]=%d, b[%d]=%d\n", i, a[i], i, b[i]);
    }

    omp_set_num_threads(4);
    printf("\nDynamic area with total amount of threads: %d\n", omp_get_max_threads());
    #pragma omp parallel for schedule(dynamic, 5)
    for (int i = 0; i < SIZE; i++)
    {
        printf("  thread №: %d, total amount of threads: %d\n", i, omp_get_thread_num(), omp_get_num_threads());
        c[i] = a[i] + b[i];
        printf("    c[%d] = %d\n", i, c[i]);
    }

    return 0;
}