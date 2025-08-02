#include <omp.h>
#include <stdio.h>

int main (int argc, char *argv[]) 
{
   int soma = 0;
   int i;

   #pragma omp parallel for schedule(static) private(soma)
   for (i = 0; i < 10000; i++)
	soma += 2;
   printf("Soma final = %d\n", soma);

   return 0;
}
