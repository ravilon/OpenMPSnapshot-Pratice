#include <stdio.h>
#include <omp.h>
#include <iostream>

const int SIZE = 40;
const int RAND_MAX_VALUE = 1000;

int main() {
    int* array = new int[SIZE];
    int count = 0;
    int mod = 9;

    srand(time(NULL));

    for (int i = 0; i < SIZE; i++) {
        array[i] = rand() % RAND_MAX_VALUE;
    }

    printf("Array initialization done. Check how it looks like: \n");
    for(int j = 0; j < SIZE; j++) {
        printf("%d ", array[j]);
    }
    printf("\n");
    

    printf("\nCalculating values that are multiples of %d.\n", mod);

    double start = omp_get_wtime();

    #pragma omp parallel shared(array, count, mod)
    #pragma omp for
    for (int i = 0; i < SIZE; i++) {
        if (array[i] % mod == 0) {
            #pragma omp atomic
            count++;
        }
    }
    
    double time = omp_get_wtime() - start;
    delete[] array;

    printf("Count = %d. \nElapsed time: %.6f\n\n", count, time);

    return 0;
}
