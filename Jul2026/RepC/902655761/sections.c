#include<stdio.h>
#include<omp.h>

int  main (){
     //omp_set_num_threads(3);
     #pragma omp parallel 
     {
          int id = omp_get_thread_num();
          #pragma omp sections nowait
          {
               #pragma  omp  section
               
                    printf("Section 0: Aula de CG - Thread %d\n", id);
               

               #pragma  omp  section
               {
                     printf("Section 1: Aula de Arquitetura - Thread %d\n", id);
     
               }

               #pragma  omp  section
               printf("Section 3: Aula de Fisica - Thread %d\n", id);

          } 
          
          printf("Fora das sections - Thread %d\n", id);
     }
     return  0;
}