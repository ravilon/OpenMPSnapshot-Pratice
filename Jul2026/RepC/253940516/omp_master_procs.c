// omp_master_procs.c
// compile with: /openmp

/* #############################################################################
## DESCRIPTION: Using pragma omp master and omp_get_num_procs().
## NAME: omp_master_procs.c
## AUTHOR: Lucca Pessoa da Silva Matos
## DATE: 10.04.2020
## VERSION: 1.0
## EXEMPLE:
##     PS C:\> gcc -fopenmp -o omp_master_procs omp_master_procs.c
##############################################################################*/

// =============================================================================
// LIBRARYS
// =============================================================================

#include <omp.h>
#include <stdio.h>
#include <locale.h>
#include <stdlib.h>

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

printf("\nNum de processadores disponivel no momento: %d", omp_get_num_procs());

printf("\n\n1 - Entrando no contexto paralelo...\n");

#pragma omp parallel
{
#pragma omp master
{
printf("\nNum de processadores disponivel no momento: %d", omp_get_num_procs());
}
}

printf("\n\n2 - Saindo do contexto paralelo...\n\n");

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
