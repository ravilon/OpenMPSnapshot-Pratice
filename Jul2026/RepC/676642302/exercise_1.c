#include <stdio.h>
#include <omp.h>

#define NUM_THREADS 4

/**
 * OpenMP program that executes on four threads, identifies the threads executing
 * and the total number of threads, and displays results on screen.
 */
int main() {
    int nThreads, threadId;

    omp_set_num_threads(NUM_THREADS);

    #pragma omp parallel shared(nThreads) private(threadId)
    {
        threadId = omp_get_thread_num();
        nThreads = omp_get_num_threads();
        printf("Hello from thread %d out of %d\n", threadId, nThreads);
    }

    return 0;
}
