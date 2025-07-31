#include <omp.h> 
#include<stdio.h>
#include <sched.h>
#define _GNU_SOURCE 
int main () 
{ 
int a[10],b[10];
int I, sum[10];
printf("Enter the values of array A :");
for(I=0;I<10;I++)
{
scanf("%d",&a[I]);
}
printf("Enter the values of array B :");
for(I=0;I<10;I++)
{
scanf("%d",&b[I]);
}
#pragma omp parallel for schedule(static, 5)//changing the number gives us varying results
for (I=0; I < 10; I++) 
{
sum[I] = a[I] + b[I]; 
printf("CPU %d \t Thread: %d Value: %d \n",sched_getcpu(),omp_get_thread_num(),sum[I]);
}
} 
