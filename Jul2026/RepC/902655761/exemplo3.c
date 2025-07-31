#include  <stdio.h>
#include  <omp.h>

int  main(){

     printf("--Fora da regiao paralela--\n");

     #pragma omp parallel
     {
        int id = omp_get_thread_num();
        int nt = omp_get_num_threads(); //Função para obter o número de threads disponível
        printf("Regiao Paralela - Hello, world da thread %d - %d threads disponiveis\n",id, nt);
     }

     printf("--Fora da regiao paralela--\n");
     return  0;
}
