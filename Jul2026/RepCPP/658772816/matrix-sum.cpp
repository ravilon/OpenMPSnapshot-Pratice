#include <omp.h>
#include <stdio.h>

#define MATRIX_DIM 50

void set_all(int val, int *mat){
        for(int i = 0; i < MATRIX_DIM; i++){
            for(int j = 0; j < MATRIX_DIM; j++){
                mat[i * MATRIX_DIM + j] = val;
            }
        }
}

int main(){

    int a[MATRIX_DIM * MATRIX_DIM];
    set_all(100, a);
    int b[MATRIX_DIM * MATRIX_DIM];
    set_all(100, b);
    int c[MATRIX_DIM * MATRIX_DIM] = {0};


    #pragma omp parallel num_threads(8) shared(c) firstprivate(a,b)
    {
        #pragma omp for collapse(2)
        for(int i = 0; i < MATRIX_DIM; i++){
            for(int j = 0; j < MATRIX_DIM; j++){
                c[i * MATRIX_DIM + j] = a[i * MATRIX_DIM + j] + b[i * MATRIX_DIM + j];
            }
        }
        
    }

    printf("%d\n", c[20]);

}