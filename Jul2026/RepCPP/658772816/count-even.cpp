#include <omp.h>
#include <stdio.h>

#define ARRAY_DIM 500

void set_all(int val, int *arr){
        for(int i = 0; i < ARRAY_DIM; i++){
                arr[i] = i;
        }
}


int main(){

    int a[ARRAY_DIM];
    set_all(100, a);
    int conteggio = 0;
    
    #pragma omp parallel shared(a)
    {
        #pragma omp for reduction(+ : conteggio)
        for(int i=0; i< ARRAY_DIM; i++){
            if(a[i] % 2 == 0){
                conteggio = conteggio + 1;
            }
        }
    }

    printf("Numeri pari =%d\n", conteggio);

}