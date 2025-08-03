#include <omp.h>
#include <stdio.h>

int main() {
#pragma omp parallel
{
#pragma omp single
{
printf("Executado pela thread %d\n", omp_get_thread_num());
}
printf("Thread %d continua execucao\n", omp_get_thread_num());
}
return 0;
}
