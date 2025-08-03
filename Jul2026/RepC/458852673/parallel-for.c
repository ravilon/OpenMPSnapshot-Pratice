#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "omp.h"

int main(){
int N = 1e6;
long int i;
double dot = 0.0;

omp_set_num_threads(4);

int *a = (int *)malloc(sizeof(int) * N);
int *b = (int *)malloc(sizeof(int) * N);

for (i = 0; i < N; i++){
a[i] = 2;
b[i] = 5;
}

clock_t start, end;
start = clock();

#pragma omp parallel for reduction(+:dot)
for(i= 0; i< N; i++){
dot += a[i] * b[i];
}

end = clock();
double time_taken = (double)(end - start) / (double)CLOCKS_PER_SEC;

free(a);
free(b);

printf("(Parallel For) Resultado: dot = %9.3f com tempo de %f segundos\n", dot, time_taken);
}