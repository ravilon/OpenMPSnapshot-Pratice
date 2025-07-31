#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define MAX_SIZE 500
int isPrime[MAX_SIZE];

void printPrimes(int limit) {
    printf("Prime numbers up to %d:\n", limit);
    if (limit >= 2) {
        printf("2 ");
    }
    for (int i = 3; i <= limit; i += 2) {
        if (isPrime[i]) {
            printf("%d ", i);
        }
    }
    printf("\n");
}

void markNonPrimes(int start, int end) {
    for (int i = start; i <= end; i += 2) {
        if (isPrime[i]) {
            int sqrtN = (int)sqrt(i);
            #pragma omp parallel for
            for (int j = 3; j <= sqrtN; j += 2) {
                if (i % j == 0) {
                    isPrime[i] = 0;
                }}}}}



int main(int argc, char *argv[]) {
    if (argc != 3)
    {
        printf("add <numProcs> <size>\n");
        return 1;
    }

    int numProcs = atoi(argv[1]);
    int size = atoi(argv[2]);
    if (size > MAX_SIZE)
     {
        printf("Size exceeds maximum limit.\n");
        return 1;
    }

    for (int i = 3; i < size; i += 2) 
    {
        isPrime[i] = 1;
    }



    double startTime = omp_get_wtime();
    #pragma omp parallel num_threads(numProcs)
    {
        int threadId = omp_get_thread_num();
        int numThreads = omp_get_num_threads();
        int blockSize = size / numThreads;
        int start = threadId * blockSize + 3;
        int end = (threadId == numThreads - 1) ? size : start + blockSize - 1;

        markNonPrimes(start, end);
    }

    double endTime = omp_get_wtime();
    printPrimes(size);
    printf("Execution time: %f seconds\n", endTime - startTime);

    return 0;
}
