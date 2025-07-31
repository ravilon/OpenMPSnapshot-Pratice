#include <stdio.h>
#include <omp.h>
#define n 15

int main(){
    int i=0;
    int a=8;
    int id=0;

	#pragma omp parallel num_threads (3) private(id)
	{
        id=omp_get_thread_num();

       #pragma omp for firstprivate(a) lastprivate(a)
        for (i=0; i<n; i++){
            a = a+1;
            printf("id=%d \t a=%d \t for i=%d\n",id,a,i);
        }
    }

    printf("Qual o valor de a?: %d ",a);
    
    return 0;
}