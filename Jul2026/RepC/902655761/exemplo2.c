#include  <stdio.h>
#include  <omp.h>

int  main(){

     printf("--Fora da regiao paralela--\n");

     #pragma omp parallel
     {
        int id = omp_get_thread_num(); //Função para obter a identificação da thread
        printf("Regiao Paralela - Hello, world da thread %d!\n",id);
     }

     printf("--Fora da regiao paralela--\n");
     return  0;
}
