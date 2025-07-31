#include <omp.h>
#include <stdio.h>
#include <conio.h>
#include <stdlib.h>
#include <time.h>

int soma_paralela(int v[]){
    int soma = 0;
    #pragma omp parallel
    {
        #pragma omp for reduction(+:soma)
        for (int i = 0; i<100; i++)
        {
            soma += v[i];
        }
    }
    return soma;
}

int soma_iterativa(int v[]){
    int soma = 0;
    for (int i = 0; i<100; i++)
    {
        soma += v[i];
    }
    return soma;
}

int main(){
    srand(time(0)); 
    omp_set_num_threads(4);

    int v[100];
    for (int i = 0; i<100; i++)
    {
        v[i] = rand() % 100;
    }

    printf("\nSoma Paralela: %d", soma_paralela(v));
    printf("\nSoma Iterativa: %d", soma_iterativa(v));

    return 0;
}
