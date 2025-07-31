//Produto  escalar  de  dois  vetores
//Cada  thread  calcula  sua  parcela  do  produto  escalar
//uso do critical para evitar acesso simultâneo das threads
//à variável dot

#include <stdio.h>
#include <omp.h>
#define n 777

int main(){
    int i=0;
    int dot=0, aux_dot=0;
    int a[n];
    int b[n];

    for (i = 0; i < n; i++){
        a[i]=1;
        b[i]=1;
    }

    #pragma omp parallel private(aux_dot)
    {
        #pragma omp for
        for (i = 0; i < n; i++){
            aux_dot += a[i]*b[i];
        }

        #pragma omp critical
        dot += aux_dot;
    }

    printf("Dot: %d ", dot);
    return 0;
}