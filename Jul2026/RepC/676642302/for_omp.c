#include <stdio.h>
#include <omp.h>

#define CHUNKSIZE 2
#define N 10

int main() {
    int i, nthreads, tid;

    int a[N], b[N], c[N];
    // Some initializations
    for (i = 0; i < N; i++) {
        a[i] = b[i] = i;
    }

    // Variables a, b & c are shared
    // Variables i & tid are private to each thread
    #pragma omp parallel shared(a, b, c) private(i, tid)
    {
        nthreads = omp_get_num_threads();

        // Iterations are statically assigned,
        // block size is fixed to chunk
        #pragma omp for schedule(static, CHUNKSIZE)
        for (i = 0; i < N; i++) {
            tid = omp_get_thread_num();
            c[i] = a[i] + b[i];
            printf("Thread %d out of %d computes iteration: %d\n", tid, nthreads, i + 1);
        }
    }   // End of the parallel region

    return 0;
}
