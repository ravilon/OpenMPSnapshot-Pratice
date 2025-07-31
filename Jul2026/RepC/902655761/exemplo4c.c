#include  <stdio.h>
#include  <omp.h>

int  main(){
   #pragma omp parallel num_threads(5)
   {
      int id = omp_get_thread_num();
      int nt = omp_get_num_threads();
      printf("Regiao Paralela 0 - Thread %d de %d threads disponiveis\n",id, nt);
   }

   #pragma omp parallel
   {
      int id = omp_get_thread_num();
      int nt = omp_get_num_threads();
      printf("Regiao Paralela 1 - Thread %d de %d threads disponiveis\n",id, nt);
   }

   omp_set_num_threads(7);
   #pragma omp parallel
   {
      int id = omp_get_thread_num();
      int nt = omp_get_num_threads();
      printf("Regiao Paralela 2 - Thread %d de %d threads disponiveis\n",id, nt);
   }

   return  0;
}