// omp_master_exemple.c
// compile with: /openmp

/* #############################################################################
## DESCRIPTION: Using pragma omp master.
## NAME: omp_master_exemple.c
## AUTHOR: Lucca Pessoa da Silva Matos
## DATE: 10.04.2020
## VERSION: 1.0
## EXEMPLE:
##     PS C:\> gcc -fopenmp -o omp_master_exemple omp_master_exemple.c
##############################################################################*/

// =============================================================================
// LIBRARYS
// =============================================================================

#include <omp.h>
#include <stdio.h>
#include <locale.h>
#include <stdlib.h>

// =============================================================================
// MACROS
// =============================================================================

#define NUM_THREADS 8

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

  printf("\nTotal de Threads a serem setadas: %d\n", NUM_THREADS);

  omp_set_num_threads(NUM_THREADS);

  printf("\nO Max de Threads eh: %d", omp_get_max_threads());

  printf("\n\n1.1 - Entrando no contexto paralelo...\n\n");

  #pragma omp parallel
  {
    int num_threads = omp_get_num_threads();
    int thread_id = omp_get_thread_num();
    #pragma omp master
    {
      printf("Thread Master %d - Max de Threads: %d\n", thread_id, omp_get_max_threads());
    }
    printf("Eu sou a Thread %d de um total de %d\n", thread_id, num_threads);
  }

  printf("\n1.2 - Saindo no contexto paralelo...\n");

  printf("\nO Max de Threads eh: %d", omp_get_max_threads());

  printf("\n\n2.1 - Entrando no contexto paralelo...\n\n");

  #pragma omp parallel num_threads(4)
  {
    int thread_id = omp_get_thread_num();
    #pragma omp master
    {
      printf("Thread Master %d - Max de Threads: %d\n", thread_id, omp_get_max_threads());
    }
  }

  printf("\n2.2 - Saindo no contexto paralelo...\n");

  printf("\nO Max de Threads eh: %d\n\n", omp_get_max_threads());

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
