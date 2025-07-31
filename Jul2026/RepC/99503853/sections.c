#include<stdio.h>
#include <omp.h>

int main()
{
     #pragma omp sections 
     {
          /*Different work can ne assigned to different thread using section*/
          #pragma omp section
          {
               printf("Hello World One\n");
          }
          #pragma omp section
          {
               printf("Hello World Two\n");
          }
          #pragma omp section
          {
               printf("Hello World Three\n");
          }
     }
}
