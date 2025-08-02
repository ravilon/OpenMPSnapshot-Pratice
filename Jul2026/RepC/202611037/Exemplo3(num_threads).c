/*
export OMP_NUM_THREADS=8 //Definir a variável de ambiente OMP_NUM_THREADS como 8. OBS: a variável de ambiente morre ao desligar o pc. 
A preferência para o número de threads é:
1° a cláusula num_threads(4).
2° a função omp_set_num_threads(6).
3° a variável de ambiente OMP_NUM_THREADS=8.
OBS: caso nenhum esteja definido rodará com uma única thread.

compilação: gcc -o openMP openMP.c -fopenmp
execução: ./openMP
*/

#include <stdio.h>  
#include <omp.h>  
#include <unistd.h>

int main() { 
   /*int i;*/ 
   printf("Fora = %d\n", omp_in_parallel()); /*omp_in_parallel retorna 0 se chamada fora da região paralela e diferente de 0 casos eja chamada dentro.*/
   omp_set_num_threads(6);

   /*default(none) faz com que o default deixe de ser shared*/
   #pragma omp parallel num_threads(4) default(none) /*shared(i)*/
   {  
      int i = omp_get_thread_num(); /*Todas as varáveis declaradas na região paralela são privadas.*/
      printf("Olá da thread %d\n", i);
      printf("Dentro = %d\n", omp_in_parallel( )); 
      printf("Número de threads = %d\n",omp_get_num_threads());
   }  
}
