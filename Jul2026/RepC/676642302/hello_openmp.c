#include <stdio.h>
#include <omp.h>

int main(void) {
    int nthreads, tid;

    #pragma omp parallel private(tid)   // Expands a group of threads
    // Each thread contains its own copy of variables
    {
        tid = omp_get_thread_num(); // Obtains the thread's identifier
        nthreads = omp_get_num_threads();   // Obtains the total number of threads
        printf("Hello world from the thread %d out of %d threads\n", tid, nthreads);
        // Every thread joins the master one and they finish
    }
    
    return 0;
}
