//Produto  escalar  de  dois  vetores
//Cada  thread  calcula  sua  parcela  do  produto  escalar
//e armazena na sua cópia da variável dot.
//No final da região paralela, a cópia original
//da variável dot é atualizada com a soma dos valores
//calculados por todas as threads.

#include <stdio.h>
#include <omp.h>
#define n 80

int main(){
    int i=0;
    int dot=0;
    int a[n];
    int b[n];

    for (i = 0; i < n; i++){
        a[i]=1;
        b[i]=1;
    }

	#pragma omp parallel
    {
        #pragma omp for private(i) reduction(+:dot)
        for (i = 0; i < n; i++){
            dot += a[i]*b[i];
        }
    }
    printf("Dot: %d \n\n", dot);
    return 0;
}