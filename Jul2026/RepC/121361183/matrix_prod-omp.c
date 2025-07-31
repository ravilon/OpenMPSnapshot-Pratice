#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include<sys/time.h>
#include"timeprint.c"
#define dim 1000
double mata[dim][dim],matb[dim][dim],matc[dim][dim];
int main(int argc,char **argv)
{
int i,j,thread;
FILE *fp;
struct timeval t1,t2;
thread=atoi(argv[1]);
for(i=0;i<dim;i++)
for(j=0;j<dim;j++)
{
mata[i][j]=rand();
matb[i][j]=rand();
}
gettimeofday(&t1,NULL);
omp_set_num_threads(thread);
#pragma omp parallel for private(j)
for(i=0;i<dim;i++)
for(j=0;j<dim;j++) matc[i][j]=mata[i][j]*matb[i][j];
gettimeofday(&t2,NULL);
fp=fopen("timp.dat","a");
timeprint(t1,t2,1,1000,fp,thread);
close(fp);
}
