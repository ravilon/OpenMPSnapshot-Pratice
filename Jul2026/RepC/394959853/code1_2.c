#include <omp.h> 
#include<stdio.h>
#include <sched.h>
#define _GNU_SOURCE 
int main () 
{
    int cube;
    int a[]={1,2,3,5,6,8};
    #pragma omp parallel for lastprivate(cube)
    for(int j=0;j<sizeof(a)/sizeof(a[0]);j++)
    {
        //printf("Thread: %d; ID: %d\n",omp_get_thread_num(),i);
        cube=a[j]*a[j]*a[j];
        printf("Cube value = %d\n",cube);
    } 
    printf("%d is the max value\n", cube);
}
