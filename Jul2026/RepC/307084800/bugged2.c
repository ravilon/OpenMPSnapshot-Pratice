/******************************************************************************
* ЗАДАНИЕ: bugged2.c
* ОПИСАНИЕ:
*   Еще одна программа на OpenMP с багом.
******************************************************************************/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    int nthreads, i, tid; // i и tid должны быть не общими
    float total = 0.0;

#pragma omp parallel private (i, tid)
    {
        // мб если количество потоков в начале должно печататься так лучше
        #pragma omp single
        {
            nthreads = omp_get_num_threads();
            printf("Number of threads = %d\n", nthreads);
        }
        tid = omp_get_thread_num();
//        if (tid == 0)
//        {
//            ntheads = omp_get_num_threads();
//            printf("Number of threads = %d\n", nthreads);
//        }
        printf("Thread %d is starting...\n", tid);

// #pragma omp barrier
        // кто - то из потоков может обнулить уже почитанное
        // поэтому надо в строке 14 инициализировать переменную
        //total = 0.0;
//#pragma omp for schedule(dynamic, 10) - ошибка, пишут одновременно в
// одну переменную несколько потоков
#pragma omp for reduction(+:total)
        for (i = 0; i < 100; i++)
            total += i*1.0;
        printf ("Thread %d is done! Total= %f\n", tid, total);
    }
}