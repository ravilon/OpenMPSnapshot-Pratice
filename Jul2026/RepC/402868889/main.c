#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int nthreads, tid;
    
    // parallel region start
    #pragma omp parallel private(nthreads, tid)
    {
        // getting thread number
        tid = omp_get_thread_num();
        printf("Hello from thread = %d\n", tid);
        
        if (tid == 0) {
            // master thread function
            nthreads = omp_get_num_threads();
            printf("Number of threads = %d\n", nthreads);
        }
    }
    
    return 0;
}