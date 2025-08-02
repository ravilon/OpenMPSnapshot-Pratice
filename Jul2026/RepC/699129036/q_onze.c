#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

void Read_matrix(char prompt[], double A[], int m, int n) {
    int i, j;

    printf("Enter the matrix %s\n", prompt);
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++) scanf("%lf", &A[i * n + j]);
}

void Read_vector(char prompt[], double x[], int n) {
    int i;

    printf("Enter the vector %s\n", prompt);
    for (i = 0; i < n; i++) scanf("%lf", &x[i]);
}

int main(int argc, char* argv[]) {
    double* A = NULL;
    double* x = NULL;
    double* y = NULL;
    int m, n;

    Get_dims(&m, &n);
    A = malloc(m * n * sizeof(double));
    x = malloc(n * sizeof(double));
    y = malloc(m * sizeof(double));
    if (A == NULL || x == NULL || y == NULL) {
        fprintf(stderr, "Erro de alocação de memória\n");
        exit(-1);
    }
    Read_matrix("A", A, m, n);
    Read_vector("x", x, n);

    int i, j;

#pragma omp parallel for num_thread(thread_count) default(none) private(i, j) \
    shared(A, x, y, m, n)
    for (i = 0; i < m; i++) {
        y[i] = 0.0;
        for (j = 0; j < n; j++) {
            y[i] += A[i * n + j] * x[j];
        }
    }
    return 0;
}