#include <stdio.h>
#include <omp.h>

int main()
{
  int N= 10, i= -1;
  #pragma omp parallel
  {
    int id= omp_get_thread_num();

    #pragma omp for schedule(static,6)//clause
    for(int i= 0; i < N; i++){
      printf("Thread %d: %d\n",id,i);
      //a[i]= b[i] + c[i];
    }

  }
}
