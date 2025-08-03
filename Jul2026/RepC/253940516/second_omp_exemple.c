// second_omp_exemple.c
// compile with: /openmp

/* #############################################################################
## DESCRIPTION: Simple exemple to read a name and show your Threads - OpenMp.
## NAME: second_omp_exemple.c
## AUTHOR: Lucca Pessoa da Silva Matos
## DATE: 10.04.2020
## VERSION: 1.0
## EXEMPLE:
##     PS C:\> gcc -fopenmp -o second_omp_exemple second_omp_exemple.c
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

#define NAME_SIZE 256

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

int thread_id, num_threads;

// Alocando.
char *name = malloc(NAME_SIZE);

// Verificando.
if (name == NULL){
printf("Sorry... We dont have memory :(\n");
return 1;
}

printf("\nHey coder! What's your name? ");
scanf("%[^\n]s",name);

printf("\nHello %s. Nice to meet you.\n", name);

printf("\n1 - We are out of the parallel context.\n\n");

// Fork
#pragma omp parallel
{
thread_id = omp_get_thread_num();
num_threads = omp_get_num_threads();
printf("Hey %s! I'm Thread %d - Total %d!\n", name, thread_id, num_threads);
}
// Join

printf("\n2 - We are out of the parallel context.\n\n");

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
