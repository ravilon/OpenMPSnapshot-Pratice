#include <omp.h>
#include <stdio.h>

#define MATRIX_DIM 1000

void set_all(int val, int *mat){
        for(int i = 0; i < MATRIX_DIM; i++){
            for(int j = 0; j < MATRIX_DIM; j++){
                mat[i * MATRIX_DIM + j] = val;
            }
        }
}

int main(){

    int a[MATRIX_DIM * MATRIX_DIM];
    set_all(10, a);
    int b[MATRIX_DIM * MATRIX_DIM];
    set_all(10, b);
    int c[MATRIX_DIM * MATRIX_DIM] = {0};

    #pragma omp parallel firstprivate(a,b) shared(c) num_threads(8)
    {
        #pragma omp for collapse(3)
        for(int i = 0; i < MATRIX_DIM; i++){
            for(int j = 0; j < MATRIX_DIM; j++){
                for(int k = 0; k < MATRIX_DIM; k++){
                    c[i * MATRIX_DIM + j] += a[i * MATRIX_DIM + k] * b[k * MATRIX_DIM + j];
                }
            }
        }
        
    }

    printf("%d\n", c[0]);

}