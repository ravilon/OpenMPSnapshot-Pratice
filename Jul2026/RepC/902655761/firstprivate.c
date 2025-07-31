#include <stdio.h>
#include <omp.h>
#define n 12

int main(){
    int i=0;
    int a=999;
    int id=0;

	#pragma omp parallel num_threads(4) firstprivate(id,a) //a entra com 999 de valor
	{
        id=omp_get_thread_num();

        #pragma omp for
        for (i=0; i<n; i++){
            a = a+1;
            printf("id=%d \t a=%d \t for i=%d\n",id,a,i);
        }
    }

    printf("Qual o valor de a?: %d ",a);

	return 0;
}