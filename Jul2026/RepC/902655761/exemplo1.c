#include  <stdio.h>
#include  <omp.h> //Biblioteca para funções OMP

int main(){
     //fork
     #pragma omp parallel     //pragma para criação da região paralela
          printf("Hello, world!\n");
     return  0;
}
