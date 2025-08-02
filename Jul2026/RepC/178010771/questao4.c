//gcc -fopenmp questao4.c -o questao4
//export OMP_NUM_THREADS=100
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define NUM_THREADS 100

int numPrime(int n){
   int k, j, i=3, soma=2;

   #pragma omp parallel shared(n) private(k,j) //variáveis k, j e i serão privadas para cada thread
    //serão criada copias privadas da variavel soma para cada threads que ao final da execução
    // do bloco serão combinadas ao operador e o resultado é colocado de volta no valor original da variável de redução compartilhada.   
    #pragma omp for reduction(+:soma) 
    for(k = 2; k < n; k++ ){
      for(j = 2; j < i; j++){
        if(i%j==0)
          break;
      }
        if(j==i){
          soma+=i;
        }
      i++;
    }
    return soma;
}

int main(int argc, char const *argv[])
{
	
    int n, soma=2;

    printf("Digite um numero:\n");
    scanf("%d",&n);

    #pragma omp parallel num_threads(NUM_THREADS)
    {    
     soma = numPrime(n);
    }
   
    printf("%d\n",soma);
   
    return 0;
}
