/*Implementação Fibonacci sequencial e recursivo
  - Gabriel Alves Mazzuco
  - Universidade Estadual do Oeste do Paraná*/

//O tamanho do fibonacci é passado via makefile

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

//Assinatura das funções
unsigned long long fib(unsigned long long n);
void fib_sequencial(int fib_tam);
void fib_parrallel(int fib_tam, int thread);

int main(int argc, char *argv[]){
  //Recebe o tamanho do fibonacci por entrada no terminal
  int fib_tam = strtol(argv[1], NULL, 10);
  int thread = strtol(argv[2], NULL, 10);

  printf("Sequencia de Fibonacci de 1 até %d\n\n", fib_tam);

  //Chama a função sequencial
  printf("Execução sequencial:\n");
  fib_sequencial(fib_tam);

  //Chama a função paralela
  printf("\n\nExecução paralela:\n");
  fib_parrallel(fib_tam, thread);

  printf("\n\n"); 

  return EXIT_SUCCESS;
}

unsigned long long fib(unsigned long long n){
  //Teste de FIB(0,1) pois os valores serão sempre 1
  if (n < 2)
      return 1;
  /*Chama a função da recursão de fibonacci, junto com a função fib_parrallel
    está será uma área de uma árvore de threads que ficarão responsáveis pela
    a sua execução*/
  else
    return fib(n-2) + fib(n-1);
}

/*A alguns modos de seleção mas é por meio dos comentários, já que o objetivo
  destre programa é sabermos o tempo de execução, não é necessário a validação
  dos valores, mas caso precise, apenas descomentar as partes que serão necessarias
  - printf = printara no terminal as respostas de FIB(n) [Consome tempo de usuário]
  - fprintf = salvará num arquivo chamado "fib.txt" as repostas de FIB(n)*/

/*Função sequencial do fibonacci, é uma implementação normal da recursão e no
  seu fim dará o tempo de execução dentro do laço que chamara a recursão*/
void fib_sequencial(int fib_tam){
  float time_sequencial = 0;
  unsigned long long int n = 0;

  //FILE *F;
  //F = fopen("fib.txt", "w");
  //fprintf(F, "Fibonacci Sequencial\n============================\n,");

  time_sequencial = omp_get_wtime();
  for (n = 0; n <= fib_tam; n++) {
      fib(n);
      //printf("Fib(%llu): %llu\n", n, fib(n));
      //fprintf(F, "FIB(%llu) = %llu\n", n, fib(n));
  }

  //fclose(F);

  time_sequencial = omp_get_wtime() - time_sequencial;
  printf("Time sequencial: %f s\n", time_sequencial);
}

/*Função paralela do fibonacci, é uma implementação normal da recursão mas
  como um adicional, existe a paralelização recursiva, e no seu fim dará o 
  tempo de execução dentro do laço que chamara a recursão*/
void fib_parrallel(int fib_tam, int thread){
  float time_parallel = 0;
  unsigned long long int n = 0;

  //FILE *F;
  //F = fopen("fib.txt", "ab");
  //fprintf(F, "\n\nFibonacci Paralela\n============================\n,");

  omp_set_num_threads(thread);

  time_parallel = omp_get_wtime();
  #pragma omp parallel private(n)
  {
    #pragma omp for schedule(dynamic, 1)
    for (n = 0; n <= fib_tam; n++){
      //printf("calculando fib de %lld na thread %d\n", n, omp_get_thread_num());
      fflush(stdout);
      fib(n);
      //printf("Fib(%llu): %llu\n", n, fib(n));
      //fprintf(F, "FIB(%llu) = %llu\n", n, fib(n));
    }
  }

  //fclose(F);

  time_parallel = omp_get_wtime() - time_parallel;
  printf("Time paralelo: %f s\n", time_parallel);
}
