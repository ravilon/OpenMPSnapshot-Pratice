#ifndef _MULTISORTH_
#define _MULTISORTH_

#include <omp.h>

#include "merge.h"      //for merge() and compare_function used in qsort()

void multisort(int *array, int *space, int N)
{
        int quarter = N/4;

        if(quarter<4)  //if there are too few elements to work with threads, use quicksort
        {
                qsort(&array[0], N, sizeof(int), compare_function);
        }
        else
        {
                int *startA = array;            
                int *spaceA = space;
                int *startB = startA + quarter; 
                int *spaceB = spaceA + quarter;
                int *startC = startB + quarter; 
                int *spaceC = spaceB + quarter;
                int *startD = startC + quarter; 
                int *spaceD = spaceC + quarter;
                
                //split the input array to 4 subarrays
                #pragma omp task
                        multisort(startA, spaceA, quarter);

                #pragma omp task
                        multisort(startB, spaceB, quarter);

                #pragma omp task
                        multisort(startC, spaceC, quarter);

                #pragma omp task
                        multisort(startD, spaceD, n - 3 * quarter);

                #pragma omp taskwait

                //merge the 4 subarrays in couples
                #pragma omp task
                        merge(&array[0], N/2, &space[0]);
                #pragma omp task
                        merge(&array[N/2], N/2, &space[N/2]);

                #pragma omp taskwait

                //merge the couples together
                #pragma omp task
                        merge(startA, startA + quarter - 1, startB, startB + quarter - 1, spaceA);

                #pragma omp task
                        merge(startC, startC + quarter - 1, startD, array + n - 1, spaceC);

                #pragma omp taskwait

                merge(spaceA, spaceC - 1, spaceC, space + n - 1, array);
        }
}

#endif
