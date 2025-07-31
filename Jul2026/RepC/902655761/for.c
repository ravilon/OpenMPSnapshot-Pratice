#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define size 64

void inicializa(int *v){
     for (int i = 0; i < size; i++)
     {
         v[i] = rand() % 100;
     }
}

int main(){
	srand(time(NULL));

	int vetor[size];

    //isso será feito sequencial
	inicializa(vetor);

	#pragma omp parallel num_threads(8)
    {
        #pragma omp for
        for(int i = 0; i < size; i++){
           int id = omp_get_thread_num();
           vetor[i] = sqrt(vetor[i]);
           printf("Thread %d - posicao %i\n", id, i);
        }
    }

	return 0;
}