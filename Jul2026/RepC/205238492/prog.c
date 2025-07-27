#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "timer.h"

int randInt(int min, int max)
{
        return (rand() % (max - min + 1)) + min;
}

int * randIntArray(unsigned int length, int min, int max)
{
        int *array;
        unsigned int i;

        array = malloc(sizeof(*array) * length);
        for (i = 0; i < length; ++i) {
                array[i] = randInt(min, max);
        }

        return array;
}

void printIntArray(int *array, unsigned int length, FILE* stream)
{
        unsigned int i;
        for (i = 0; i < length; ++i) {
                fprintf(stream, "%02d ", array[i]);
        }
}

void printlnIntArray(int *array, unsigned int length, FILE* stream)
{
        printIntArray(array, length, stream);
        fprintf(stream, "\n");
}

int main(int argc, char const *argv[])
{
        int numThreads;
        unsigned int length;
        int minRand;
        int maxRand;
        double mean;
        int moduloProduct;
        int minimumValue;
        int maximumValue;
        int *values;
        double inicioTotal, fimTotal, tempoTotal;
        double inicio[4], fim[4], tempo[4];
        unsigned int i;

        if (argc < 3) {
                printf("Error missing command line argument.\n");
                return 1;
        }

        length = atoi(argv[1]);
        minRand = atoi(argv[2]);
        maxRand = atoi(argv[3]);
        numThreads = atoi(argv[4]);

        values = randIntArray(length, minRand, maxRand);


        GET_TIME(inicioTotal);
        #pragma omp parallel num_threads(numThreads) private(i)
        {
                #pragma omp sections nowait
                {
                        #pragma omp section
                        {
                                GET_TIME(inicio[0]);
                                mean = 0;
                                for (i = 0; i < length; ++i) {
                                        mean += values[i];
                                }
                                mean /= (double)length;
                                GET_TIME(fim[0]);
                                tempo[0] = fim[0] - inicio[0];
                        }

                        #pragma omp section
                        {
                                GET_TIME(inicio[1]);
                                minimumValue = maxRand;
                                for (i = 0; i < length; ++i) {
                                        if (values[i] < minimumValue) {
                                                minimumValue = values[i];
                                        }
                                }
                                GET_TIME(fim[1]);
                                tempo[1] = fim[1] - inicio[1];
                        }

                        #pragma omp section
                        {
                                GET_TIME(inicio[2]);
                                maximumValue = minRand;
                                for (i = 0; i < length; ++i) {
                                        if (values[i] > maximumValue) {
                                                maximumValue = values[i];
                                        }
                                }
                                GET_TIME(fim[2]);
                                tempo[2] = fim[2] - inicio[2];
                        }

                        #pragma omp section
                        {
                                GET_TIME(inicio[3]);
                                moduloProduct = 1;
                                for (i = 0; i < length; ++i) {
                                        moduloProduct = (moduloProduct * ((int)pow(values[i], 2.0))) % 10000007;
                                }
                                GET_TIME(fim[3]);
                                tempo[3] = fim[3] - inicio[3];
                        }
                }
        }

        GET_TIME(fimTotal);
        tempoTotal = fimTotal - inicioTotal;

        // for (i = 0; i < 4; ++i) {
        //         printf("Tempo tarefa %d: %.8lf\n", i, tempo[i]);
        // }
        printf("Tempo Total: %.8lf\n", tempoTotal);

        // printf("Mean: %02.2lf\n", mean);
        // printf("Min: %d\n", minimumValue);
        // printf("Max: %d\n", maximumValue);
        // printf("Modulo Product: %d\n", moduloProduct);

        free(values);

        return 0;
}