#include <stdio.h>
#include <omp.h>
#define N 100

int main()
{
int num_t, i, suma = 0; 

num_t = omp_get_num_procs();
omp_set_num_threads(num_t);

// Sumamos los N elementos desde el 1 hasta N.
#pragma omp parallel for reduction(+:suma) 
for (i = 1; i <= N; i++) 
{
suma += i;
}

printf("El resultado de la suma de 1 hasta %d es: %d\n", N, suma);

return 0;
}
