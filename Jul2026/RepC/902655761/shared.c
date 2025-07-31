#include <stdio.h>
#include <omp.h>
#define size 20

void inicializa(int *v){
    for (int i = 0; i < size; i++){
        v[i] = 0;
    }
}

int main(){
	int vetor[size];

	inicializa(vetor);

	#pragma omp parallel for num_threads(3) shared (vetor)
    for(int i = 0; i < size; i++){
        int id = omp_get_thread_num();
        vetor[i] = vetor[i]+id;
    }

    for (int i = 0; i < size; i++){
        printf("V[%d]: %d\n", i, vetor[i]);
    }

	return 0;
}