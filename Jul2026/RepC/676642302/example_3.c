#include <stdio.h>
#include <omp.h>

int main() {
    int nthreads, tid;
    omp_set_num_threads(4);

    // The `tid` variable is private to each thread
    #pragma omp parallel private(tid)
    {
        nthreads = omp_get_num_threads();
        
        #pragma omp sections
        {
            #pragma omp section
            {
                tid = omp_get_thread_num();
                printf("Thread %d out of %d computes section 1\n", tid, nthreads);
            }

            #pragma omp section
            {
                tid = omp_get_thread_num();
                printf("Thread %d out of %d computes section 2\n", tid, nthreads);
            }

            #pragma omp section
            {
                tid = omp_get_thread_num();
                printf("Thread %d out of %d computes section 3\n", tid, nthreads);
            }

            #pragma omp section
            {
                tid = omp_get_thread_num();
                printf("Thread %d out of %d computes section 4\n", tid, nthreads);
            }
        }   // end of sections
    }   // end of the parallel section
    return 0;
}
