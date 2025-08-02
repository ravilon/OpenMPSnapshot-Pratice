#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "omp.h"

#define NUM_THREADS 4

float f(float x);
float calculaIntegral(float *limits, float deltaX, int nTrapezoidesPorRank);

int main(int argc, char **argv){
    clock_t begin = clock();

    omp_set_num_threads(NUM_THREADS);

    int limitA = atoi(argv[1]);
    int limitB = atoi(argv[2]);
    int nTrapezoides = atoi(argv[3]);

    float soma = 0.0;

    float deltaX = (float) (limitB - limitA)/nTrapezoides;

    int nTrapezoidesPorRank = nTrapezoides / NUM_THREADS;

    double *parcialSum = (double *)calloc(NUM_THREADS, sizeof(double));

    #pragma omp parallel shared(parcialSum, limitA, limitB, nTrapezoidesPorRank)
    {
        float a, b;

        int id, nThreads;
        double x;

        id = omp_get_thread_num();
        nThreads = omp_get_num_threads();

        a = limitA + (id * nTrapezoidesPorRank * deltaX);
        b = (id == nThreads - 1) ? limitB : a + (nTrapezoidesPorRank * deltaX);

        if(nTrapezoides % nThreads > 0){
            if(id < (nTrapezoides % nThreads)){
                nTrapezoidesPorRank++;
                a = (a + (id * nTrapezoidesPorRank * deltaX));
                b = (a + nTrapezoidesPorRank * deltaX);
            } else {
                a = (a + id * nTrapezoidesPorRank * deltaX) + ((nTrapezoides % nThreads) * deltaX);
                b += (id == nThreads - 1) ? 0 : ((nTrapezoides % nThreads) * deltaX);
            }
        }

        float limits[2] = { a, b };

        parcialSum[id] = calculaIntegral(limits, deltaX, nTrapezoidesPorRank);

    }

    // Redução
    for(int i = 0; i < NUM_THREADS; i++){
        soma += parcialSum[i];
    }

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    printf("INTEGRAL: %f\nTIME: %f\n", soma, time_spent);
    
    return 0;
}

float f(float x){
    return x * x;
}

float calculaIntegral(float *limits, float deltaX, int nTrapezoidesPorRank){
    float soma = (f(limits[0]) + f(limits[1]))/2.0;
    float x = limits[0];

    for(int i = 0; i < nTrapezoidesPorRank; i++){
        x += deltaX;
        soma += f(x);
    }

    return soma * deltaX;
}