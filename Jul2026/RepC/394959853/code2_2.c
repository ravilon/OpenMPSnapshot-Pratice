#include <omp.h> 
#include<stdio.h>
#include <sched.h>
#define _GNU_SOURCE 
int main () 
{
    int i=10;
    #pragma omp parallel for firstprivate(i)
    for(int j=0;j<10;j++)
    {
        i=i+j;
        printf("Thread: %d; ID: %d\n",omp_get_thread_num(),i);
        printf("i = %d\n",i);
    } 
}
