#include <stdio.h>
#include <omp.h>

#define N 10

int main() 
{
int matrizA[N][N];

#pragma omp parallel for collapse(2)
for (int i = 0; i < N; i++)
{
for (int j = 0; j < N; j++)
{
matrizA[i][j] = i+j; 
}
}

printf("Matriz A:\n");
for (int i = 0; i < N; i++)
{
for (int j = 0; j < N; j++)
{
printf("%d\t", matrizA[i][j]); 
}
printf("\n");
}

return 0;
}
