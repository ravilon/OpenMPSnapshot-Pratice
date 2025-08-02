
#include <stdio.h>
#include <omp.h>

int main(){
    double time=omp_get_wtick();
    #pragma omp parallel
    {
        printf("Hello from Thread %d\n",omp_get_thread_num());
    }

    printf("\n----------------------------\n");
    printf("Minimum time that can be measured using wtime() is %fs",time);
    printf("\n----------------------------\n");
}
