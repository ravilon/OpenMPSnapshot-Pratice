#include <stdio.h>
#include <omp.h>

int main() {
    const int numThreads = 4;
    const int numIterations = 5;
    int runningTotal = 0;
    printf("Done by 2021BCS0084 Dhivyesh RK\n");
    #pragma omp parallel num_threads(numThreads)
    {
        int threadID = omp_get_thread_num();
        int localSum = 0;

        for (int i = 0; i < numIterations; ++i) {
	    //1st Barrier - all threads will synchronise here before localsum update
            #pragma omp barrier

            localSum += threadID * (i + 1);

            // 2nd Barrier - All threads synchronize here before updating runningTotal
            #pragma omp barrier

            // Update runningTotal after Barrier 2
            #pragma omp critical
            {
                runningTotal += localSum;
            }

            // Barrier 3: All threads synchronize here before moving to the next iteration
            #pragma omp barrier

            // Print the progress for each thread
            printf("Thread %d - Iteration %d: localSum = %d, runningTotal = %d\n",
                   threadID, i, localSum, runningTotal);
        }
    }

    printf("Final runningTotal: %d\n", runningTotal);

    return 0;
}

