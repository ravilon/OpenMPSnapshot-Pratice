
#include <stdio.h>
#include <omp.h>

int main(){
    int shared_variable=0;
    #pragma omp parallel
    {
        int thread_id=omp_get_thread_num();
        shared_variable+=thread_id;
        printf("Shared variable: %d from thread %d\n",shared_variable,thread_id);
    }

    printf("Final outcome of shared variable: %d",shared_variable);
}
