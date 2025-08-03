
#include<stdio.h>
#include<omp.h>

int main()
{
#pragma omp parallel
{
#pragma omp sections
{
// Each section is executed by one of the threads
#pragma omp section
{
printf("Section 1 for thread %d\n",omp_get_thread_num());
}
#pragma omp section
{
printf("Section 2 for thread %d\n",omp_get_thread_num());
}
#pragma omp section
{
printf("Section 3 for thread %d\n",omp_get_thread_num());
}
#pragma omp section
{
printf("Section 4 for thread %d\n",omp_get_thread_num());
}
#pragma omp section
{
printf("Section 5 for thread %d\n",omp_get_thread_num());
}
#pragma omp section
{
printf("Section 6 for thread %d\n",omp_get_thread_num());
}
}
}
}
