#include <stdio.h>
#include <omp.h>

int main() {
    int nthreads, tid;
    printf("We will work with 4 threads\n");
    omp_set_num_threads(4);

    // Obtains the number of threads currently running
    nthreads = omp_get_num_threads();
    printf("Number of threads running before the parallel section: %d\n", nthreads);

    #pragma omp parallel private(tid)   // Expands a group of threads
    // Each thread contains its own copy of the `tid` variable
    {
        tid = omp_get_thread_num(); // Obtains the thread's identifier
        printf("Hello from thread %d\n", tid);
        if (!tid) { // This section is executed only by the master thread
            nthreads = omp_get_num_threads();
            printf("Number of threads = %d\n", nthreads);
        }
    }   // All threads join the master and finish

    printf("\nWe will work with 3 threads\n");

    omp_set_num_threads(3);
    nthreads = omp_get_num_threads();
    printf("Number of threads running before the parallel section: %d\n", nthreads);
    #pragma omp parallel private(tid)
    {
        tid = omp_get_thread_num();
        printf("Hello from thread %d\n", tid);
        if (!tid) {
            nthreads = omp_get_num_threads();
            printf("Number of threads = %d\n", nthreads);
        }
    }

    return 0;
}
