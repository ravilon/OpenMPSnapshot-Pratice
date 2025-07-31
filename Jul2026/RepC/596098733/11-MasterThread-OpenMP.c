/*
 * 11-MasterThread-OpenMP.c
 *
 *  Created on: 11 feb. 2023
 *      Author: Jose ngel Gumiel
 */


#include <stdio.h>
#include <omp.h>

int main()
{
    int nthreads, tid;

    #pragma omp parallel private(tid)
    {
        tid = omp_get_thread_num();
        #pragma omp master
        {
        	//This section of code is only executed by the master.
            nthreads = omp_get_num_threads();
            printf("I am the master, pID %d. Number of threads = %d\n", tid, nthreads);
        }
        //This section is executed by every thread, including the master.
        printf("Hello World from thread %d\n", tid);
    }
    return 0;
}
