#include <stdio.h>
#include "omp.h"

int main(void){
    omp_set_num_threads(4);

    int x = 5;

    #pragma omp parallel
    {
        x++;
    }

    printf("shared: x is %d\n", x);

    return 0;
}