#include <stdio.h>
#include <omp.h>

int main() {
    const int numIterations = 2;
    printf("Done by Dhivyeshrk\n");
    #pragma omp parallel
    {
        #pragma omp single
        {
            printf("Executing omp single - 2021BCS0084 Dhivyeshrk\n");
            for (int i = 0; i < numIterations; ++i) {
                printf("Single iteration %d \n", i);
            }
        }

        #pragma omp critical
        {
            printf("Executing omp critical \n");
            for (int i = 0; i < numIterations; ++i) {
                printf("Critical iteration %d\n", i);
            }
        }
    }

    return 0;
}

