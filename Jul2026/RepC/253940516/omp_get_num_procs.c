// omp_get_num_procs.c
// compile with: /openmp

/* #############################################################################
## DESCRIPTION: Using omp_get_num_procs() to show your CPUS.
## NAME: omp_get_num_procs.c
## AUTHOR: Lucca Pessoa da Silva Matos
## DATE: 10.04.2020
## VERSION: 1.0
## EXEMPLE:
##     PS C:\> gcc -fopenmp -o omp_get_num_procs omp_get_num_procs.c
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

#define NUM_THREADS 4

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

omp_set_num_threads(NUM_THREADS);

printf("\nQuantidade de CPU(s) disponíveis no momento: %d...\n", omp_get_num_procs());
printf("\n1 - Estamos fora do contexto paralelo...\n\n");

// Fork
#pragma omp parallel
{
int id = omp_get_num_threads();
int thread_id = omp_get_thread_num();
printf("Eu sou a Thread %d de um total de %d\n", thread_id, id);
}
// Join

printf("\n2 - Estamos fora do contexto paralelo...\n\n");
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
