#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char **argv){
  int i,n=9;
  if(argc<2){
      fprintf(stderr,"\nFalta nº de iteraciones\n");
      exit(-1);
  }
  n= atoi(argv[1]);
  #pragma omp parallel
  {
  #pragma omp for
      for(i=0;i<n;i++)
  #pragma omp parallel
	 printf("thread %d ejecuta la iteración %d del bucle\n",omp_get_thread_num(),i);
  }
  return(0);
}