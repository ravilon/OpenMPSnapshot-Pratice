#include "omp.h"
#include<stdio.h> /* OpenMP compiler directives, APIs are
declared here */
void​ ​ main​ ()
{
/* Parallel region begins here */
double​ time;
double​ start;
double​ end;
start = omp_get_wtime();
#pragma omp parallel
{
int​ id = omp_get_thread_num();
printf​ ( ​ "hello world-%d\n"​ , id);
}
end = omp_get_wtime();
printf​ ( ​ "Work took %f seconds\n"​ , end - start);
}
