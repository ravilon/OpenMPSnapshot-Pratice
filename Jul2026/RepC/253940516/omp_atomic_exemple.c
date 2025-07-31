// omp_atomic_exemple.c
// compile with: /openmp

/* #############################################################################
## DESCRIPTION: Simple exemple using atomic in OpenMP.
## NAME: omp_atomic_exemple.c
## AUTHOR: Lucca Pessoa da Silva Matos
## DATE: 10.04.2020
## VERSION: 1.0
## EXEMPLE:
##     PS C:\> gcc -fopenmp -o omp_atomic_exemple omp_atomic_exemple.c
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

int contador = 0;

printf("\n1 - Estamos fora do contexto paralelo. Entrando...\n");

#pragma omp parallel num_threads(NUM_THREADS)
{
#pragma omp atomic
contador++;
}

printf("\n2 - Estamos fora do contexto paralelo. Saindo...\n");

printf("\nNumber of threads: %d\n\n", contador);

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
