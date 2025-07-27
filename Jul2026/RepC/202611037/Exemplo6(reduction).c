#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define SUM_INIT 0
int main(int argc, char *argv[]) {
   int i, n = 25;
   int sum, a[n];
   int ref = SUM_INIT + (n-1)*n/2;
   (void) omp_set_num_threads(3);
   for (i=0; i<n; i++)    
          a[i] = i;
#pragma omp parallel
   { /*#pragma omp master*/ /*Indica que um bloco seve ser executado apenas pela thread master (thread 0). OBS: Outras threads pulam o bloco e continuam a execução: é diferente da diretiva single neste aspecto. */
      #pragma omp single /*A primeira thread que chegar nesse ponto executará o trecho de código a baixo sozinha. OBS: As demais threads esperam pela thread que está executando o bloco, para todas prosseguirem juntas.*/
          printf("Número de threads é %d\n", omp_get_num_threads());
   }
 sum = SUM_INIT;
   printf("Valor da soma antes da região paralela: %d\n",sum);

   #pragma omp parallel for default(none) shared(n,a)  reduction(+:sum)
        for (i=0; i<n; i++)
          sum += a[i];
   /*-- Fim da redução paralela  --*/

   printf("Valor da soma depois da região paralela: %d\n",sum);
   printf("Verificação do resultado: soma = %d (deveria ser %d)\n",sum,ref);
   return(0);
}

