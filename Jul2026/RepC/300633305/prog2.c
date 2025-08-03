#include "omp.h" /* OpenMP compiler directives, APIs are declared here */
#include<stdio.h>
void printHello(ID){
printf("Hello World\n");
}

int main()
{
/* Parallel region begins here */

double start; 
double end; 
start = omp_get_wtime(); 
#pragma omp parallel num_threads(4)
{
int ID = omp_get_thread_num();
printHello(ID);
}
end = omp_get_wtime(); 
printf("Work took %f seconds\n", end - start);
return 0;
}
