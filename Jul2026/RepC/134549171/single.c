/*
* In the example, only one thread prints each of the progress messages. All other threads will skip the single region and stop at the barrier at the end of the single construct until all threads in the team have reached the barrier.
*
* If other threads can proceed without waiting for the thread executing the single region, a nowait clause can be specified, as is done in the third single construct in this example.
*
* The user must not make any assumptions as to which thread will execute a single region.
*/

#include <stdio.h>

void work1() {}
void work2() {}

void single_example()
{
#pragma omp parallel
{
#pragma omp single
printf("Beginning work1.\n");
work1();
#pragma omp single
printf("Finishing work1.\n");
#pragma omp single nowait
printf("Finished work1 and beginning work2.\n");
work2();
}
}
int main()
{
single_example();
return 0;
}
