#include <omp.h>
#include <stdio.h>

#define IS_PARALLEL true

int main(){

    #pragma omp parallel if(IS_PARALLEL) num_threads(4) 
    {
        printf("Parallel code executed by %d threads\n", omp_get_num_threads());
        if(omp_get_thread_num() == 2){
            printf("Thread %d does different work\n", omp_get_thread_num());
        }
        printf("I'm thread %d\n", omp_get_thread_num());
    }
    return 0;
}