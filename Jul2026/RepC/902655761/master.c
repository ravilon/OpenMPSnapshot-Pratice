#include<stdio.h>
#include<omp.h>
#define N 32

int  main (){
     int a[N];

     #pragma  omp  parallel
     {
           int id = omp_get_thread_num();

           #pragma omp for
           for(int i=0; i<N; i++ )
              a[i] = id;

           #pragma  omp  master
              printf("Thread %d executou omp_master\n", id);

           #pragma  omp  for
           for(int i=0; i<N; i++ )
              a[i]=id*id;
     }
     return  0;
}