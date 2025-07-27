/******************************************************************************
 * FILE: omp_mm.c
 * DESCRIPTION:  
 *   OpenMp Example - Matrix Multiply - C Version
 *   Demonstrates a matrix multiply using OpenMP. Threads share row iterations
 *   according to a predefined chunk size.
 * AUTHOR: Blaise Barney
 * LAST REVISED: 06/28/05
 ******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NRA 200 /* numero de linhas na matriz A  */
#define NCA 200 /* numero de colunas na matrix A */
#define NCB 200 /* numero de linhas na matrix B  */

// Calcula o tempo entre duas tomadas de tempo em segundos
float calculaTempoSegundos(clock_t ini, clock_t fim) {
    return ((double)fim-ini)/CLOCKS_PER_SEC;
}

int main(int argc, char * argv[]) {
  clock_t ini, fim;
  int tid, nthreads;
  double a[NRA][NCA], 
    b[NCA][NCB], 	  
    c[NRA][NCB]; 	  

   ini = clock();

    /***  Inicializar matrizes ***/
    for (int i = 0; i < NRA; i++)
      for (int j = 0; j < NCA; j++)
        a[i][j] = i + j;
        
    for (int i = 0; i < NCA; i++)
      for (int j = 0; j < NCB; j++)
        b[i][j] = i * j;
        
    for (int i = 0; i < NRA; i++)
      for (int j = 0; j < NCB; j++)
        c[i][j] = 0;

    printf("Thread %d iniciando multiplicacao de matrizes \n", tid);
    
	#pragma omp parallel for
    for (int i = 0; i < NRA; i++) {
      printf("Thread %d calculou a linha %d\n", tid, i);
      for (int j = 0; j < NCB; j++)
        for (int k = 0; k < NCA; k++)
          c[i][j] += a[i][k] * b[k][j];
    }

  /*** Imprimir resultados ***
  printf("******************************************************\n");
  printf("Matriz Resultante:\n"); 
  for (int i = 0; i < NRA; i++) {
    for (int j = 0; j < NCB; j++)
      printf("%6.2f   ", c[i][j]);
    printf("\n");
  }
  printf("******************************************************\n");
  */
  fim = clock();
  printf("Tempo Insertion Sort %f\n", calculaTempoSegundos(ini,fim));
  printf("Terminou.\n");

}
