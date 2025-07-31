#include  <stdio.h>
#include  <omp.h>

int  main(){

     #pragma omp parallel num_threads(5)
     {
        int id = omp_get_thread_num();
        printf("Regiao Paralela  - Thread %d\n",id);

        #pragma omp barrier
        if (id == 0){
                  int nt = omp_get_num_threads();
                  printf("Total de threads= %d\n",nt);

        }
     }

     return  0;
}