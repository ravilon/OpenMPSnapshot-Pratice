// omp_single_exemple.c
// compile with: /openmp

/* #############################################################################
## DESCRIPTION: Simple exemple using single in OpenMP.
## NAME: omp_single_exemple.c
## AUTHOR: Lucca Pessoa da Silva Matos
## DATE: 10.04.2020
## VERSION: 1.0
## EXEMPLE:
##     PS C:\> gcc -fopenmp -o omp_single_exemple omp_single_exemple.c
##############################################################################*/

// =============================================================================
// LIBRARYS
// =============================================================================

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <locale.h>

// =============================================================================
// MACROS
// =============================================================================

#define NUM_THREADS 12

// =============================================================================
// CALL FUNCTIONS
// =============================================================================

void cabecalho();
void set_portuguese();

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char const *argv[]){

  set_portuguese();
  cabecalho();

  int thread_id;

  printf("\n1 - Estamos fora do contexto paralelo. Entrando...\n\n");

  #pragma omp parallel num_threads(2)
  {
    #pragma omp single
    // Only a single thread can read the input.
    printf("Read input\n");
    // Multiple threads in the team compute the results.
    printf("Compute results\n");
    #pragma omp single
    // Only a single thread can write the output.
    printf("Write output\n");
  }

  printf("\n2 - Estamos fora do contexto paralelo. Saindo...\n");

  return 0;
}

// =============================================================================
// FUNCTIONS
// =============================================================================

void set_portuguese(){
  setlocale(LC_ALL, "Portuguese");
}

void cabecalho(){
  printf("\n**************************************************");
  printf("\n*                                                *");
  printf("\n*                                                *");
  printf("\n* PROGRAMACAO PARALELA COM OPENMP - LUCCA PESSOA *");
  printf("\n*                                                *");
  printf("\n*                                                *");
  printf("\n**************************************************\n");
}
