#include <stdio.h>
#include <omp.h>

int main(){
    int thread_id;
    int thread_count=omp_get_num_threads();
    #pragma omp parallel num_threads(4)
    {
        thread_id=omp_get_thread_num();
        printf("Hello from thread no: %d out of %d\n",thread_id,thread_count);
    }
}
