/*
According to $, a thread must not execute more than one ordered region that binds to the
same loop region. 

So the collapse clause is required for the example to be
conforming. 

With the collapse clause, the iterations of the k and j loops are
collapsed into one loop, and therefore only one ordered region will bind to the collapsed            ------> IN THE SUB FUNCTION
k and j loop. 

Without the collapse clause, there would be two ordered regions that
bind to each iteration of the k loop (one arising from the first iteration of the j loop, and        ------> IN THE SUB-2 FUNCTION
the other arising from the second iteration of the j loop).


The sub2 functions runs fine but it is not according to openmp standards.

*/
#include <omp.h>
#include <stdio.h>

void sub()
{
int j, k, a;
#pragma omp parallel num_threads(2)
{
#pragma omp for collapse(2) ordered private(j,k) schedule(static,1)
for (k=1; k<=80; k++)
for (j=1; j<=75; j++)
{
#pragma omp ordered
printf("%d %d %d\n", omp_get_thread_num(), k, j);
}
}
}

void sub2()
{
int j, k, a;
#pragma omp parallel num_threads(2)
{
#pragma omp for ordered private(j,k) schedule(static,1)
for (k=1; k<=80; k++)
for (j=1; j<=75; j++)
{
#pragma omp ordered
printf("%d %d %d\n", omp_get_thread_num(), k, j);
}
}
}

void serial()
{
int j, k, a;

for (k=1; k<=80; k++)
for (j=1; j<=75; j++)
{
printf("%d %d %d\n", omp_get_thread_num(), k, j);
}

}


int main()
{
double start_time, run_time;

printf("Ordered clause used along with the collapse clause\n");
start_time = omp_get_wtime();
sub();
run_time = omp_get_wtime() - start_time;
printf("Time to compute(in parallel) : %f\n", run_time);

printf("Ordered clause used without the collapse clause\n");
start_time = omp_get_wtime();
sub2();
run_time = omp_get_wtime() - start_time;
printf("Time to compute(in parallel) : %f\n", run_time);

/*printf("Serially : \n");
start_time = omp_get_wtime();
serial();
run_time = omp_get_wtime() - start_time;
printf("Time to compute(in serial) : %f\n", run_time);

*/
return 0;
}

