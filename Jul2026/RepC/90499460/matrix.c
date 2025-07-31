#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>

/* initialize a matrix x. You need to call srand() in advance. */
void Mat_Init(int row, int col, double *X) {
    int i, size;

    size = row * col;
    for (i = 0; i < size; i++) {
        // X[i] = ( (double)rand() ) / ( (double)RAND_MAX);
        X[i] = (double) (rand() % 10 + 1);
    }
}

/* display a matrix */
void Mat_Show_p(int row, int col, double *X, int precision) {
    int i, j;
    printf("row = %d col = %d\n", row, col);
    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            if (precision < 0) {
                printf("%lf ", X[col * i + j]);
            } else {
                printf("%.*f ", precision, X[col * i + j]);
            }
        }
        printf("\n");
    }
}

void Mat_Show(int row, int col, double *X) {
    Mat_Show_p(row, col, X, -1);
}

void Mat_Clone(int row, int col, double *Src, double *Dest) {
    int i, size;

    size = row * col;
    for (i = 0; i < size; i++) {
        Dest[i] = Src[i];
    }
}

/* matrix vector multiplication: Y = Xv where v is a vector */
void Mat_Xv(int row, int col, double *X, double *Y, double *v) {
    int i, j;
    double result;

    for (i = 0; i < row; i++) {
        result = 0;
        for (j = 0; j < col; j++)
            result += v[j] * X[col * i + j];
        Y[i] = result;
    }
}

/* matrix vector multiplication by OpenMP: Y = Xv where v is a vector */
void Omp_Mat_Xv(int row, int col, double *X, double *Y, double *v, int thread_count) {
    int i, j;
    double result;

    /* add your OpenMP statement before the for loop */
# pragma omp parallel for num_threads(thread_count) default(none) private(i, j, result) shared(row, col, X, Y, v)
    for (i = 0; i < row; i++) {
        result = 0;
        for (j = 0; j < col; j++)
            result += v[j] * X[col * i + j];
        Y[i] = result;
    }
}