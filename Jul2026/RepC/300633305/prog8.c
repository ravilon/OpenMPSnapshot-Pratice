#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define NUM 100

// PRODUCER: initialize A with random data
void fillRandom(int nval, double *A)
{
int i;
for (i=0; i<nval; i++)
A[i] = (double) rand()/1111111111;
}
// CONSUMER: Sum the data in A
double arraySum(int nval, double *A)
{
int i;
double sum = 0.0;
for (i=0; i<nval; i++)
sum = sum + A[i];
return sum;
}


int main()
{
double *A, sum, runtime;
int flag = 0;
A = (double *)malloc(NUM*sizeof(double));
runtime = omp_get_wtime();
#pragma omp parallel shared(A,sum,flag)
{
#pragma omp sections
{
#pragma omp section
{
fillRandom(NUM,A);
#pragma omp flush
flag = 1;
if(flag==1)
{
#pragma omp flush(flag)
}
}
#pragma omp section
{
while (!flag)
{
#pragma omp flush(flag)
}
#pragma omp flush
sum = arraySum(NUM,A);
}
}
}
runtime = omp_get_wtime() - runtime;
printf("Sum is %f\n",sum);
printf("Work took %f seconds\n", runtime);
}
