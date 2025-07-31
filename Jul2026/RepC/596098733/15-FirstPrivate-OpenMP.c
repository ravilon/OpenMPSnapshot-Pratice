/*
 * 15-FirstPrivate-OpenMP.c
 *
 *  Created on: 15 feb. 2023
 *      Author: Jose ngel Gumiel
 */

#include <stdio.h>
#include <omp.h>

int main() {
    int x = 0;
    //Each thread picks the value of x and modifies it.
    #pragma omp parallel firstprivate(x)
    {
        int id = omp_get_thread_num();
        x += id;
        printf("Thread %d: x = %d\n", id, x);
    }
    //After the parallel section, x still has the initial value.
    printf("After parallel region: x = %d\n", x);
    return 0;
}

