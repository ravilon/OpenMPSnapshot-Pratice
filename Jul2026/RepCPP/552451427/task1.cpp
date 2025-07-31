#include <iostream>
#include <omp.h>

int main() {
    #pragma omp parallel num_threads(8)
    {
        int threadsCount = omp_get_num_threads();
        int threadID = omp_get_thread_num();
        printf("Hello world! Thread №: %d; Threads count: %d\n", threadID, threadsCount);
    }

    return 0;
}
