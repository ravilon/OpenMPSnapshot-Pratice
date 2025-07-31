#include <iostream>
#include <omp.h>

int main() {
    printf("1. FOR LOOP.\n");
    int max = 8;

    #pragma omp parallel num_threads(max)
    {
        #pragma omp for ordered
        for(int i = max - 1; i >= 0; i--) {
            #pragma omp ordered
            {
                int threadID = omp_get_thread_num();
                printf("Hello world! Thread №: %d; Threads count: %d\n", threadID, omp_get_num_threads());
            }
        }
        
    }

    return 0;
}
