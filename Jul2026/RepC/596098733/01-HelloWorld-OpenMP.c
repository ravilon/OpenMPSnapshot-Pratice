/*
 * 01-HelloWorld-OpenMP.c
 *
 *  Created on: 1 feb. 2023
 *      Author: Jose ngel Gumiel
 */

#include <stdio.h>
#include <omp.h>

int main() {
    #pragma omp parallel num_threads(16)
    {
        int id = omp_get_thread_num();
        printf("Hello World from thread %d\n", id);
    }

    return 0;
}
