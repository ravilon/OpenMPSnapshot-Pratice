#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<string.h>
#include<omp.h>

//gcc main.c -o main -fopenmp && ./main 1000 1000 1000 1000 o

void preencheMatriz(int **M, int l, int c){
int r=0;    
  for(int i=0;i<l;i++){
    for(int j=0;j<c;j++){
      r = (rand() % 100);
      M[i][j]=r; 
    }
  }
}

void MulM1M2(int **M1, int **M2, int **MR, int l, int c){
    int aux = 0;

    omp_set_num_threads(2);
    int i,j,k;
    #pragma omp parallel for private(aux,j,k) firstprivate(l,c) shared(i)
      for (i=0;i<l;i++){
            for (j=0;j<c;j++){
                aux = 0;
                for(k=0;k<c;k++) aux+=M1[i][k]*M2[k][j];
                
                MR[i][j] = aux;    
            }
      }
}

int main(int argc, char *argv[]){
  double tempo;

  if(argc < 6){
     printf("Insira todos os valores necessarios!\n");
     exit(-1);
  }

  else{
    int l1=0,c1=0,l2=0,c2=0;
    int **M1, **M2,**M2T,**MR;

    //pegando os valores do terminal
    l1= (int) atoi(argv[1]); //converte de ascii para int
    c1= (int) atoi(argv[2]); //converte de ascii para int
    l2= (int) atoi(argv[3]); //converte de ascii para int
    c2= (int) atoi(argv[4]); //converte de ascii para int
    
    //Alocando as matrizes
    //Alocando as linhas da matriz:
    M1= malloc(l1 * sizeof(int*));
    M2= malloc(l2 * sizeof(int*));
    MR= malloc(l1 * sizeof(int*));

   //Alocando as colunas da matriz:
    for(int i=0; i<l1; i++){
      M1[i]= malloc(c1 * sizeof(int));
      MR[i]= malloc(c1 * sizeof(int));
    }
    for(int i=0; i<l2; i++){
      M2[i]= malloc(c2 * sizeof(int));
    }

    //Atribuindo valores para a matriz
    preencheMatriz(M1,l1,c1);
    preencheMatriz(M2,l2,c2);

    //Matriz M1
    /*for(int i=0;i<l1;i++){
      for(int j=0;j<c1;j++)
          printf("%d ",M1[i][j]);
      printf("\n");
    }

    //Matriz M2
    printf("\n");
    for(int i=0;i<l2;i++){
      for(int j=0;j<c2;j++)
          printf("%d ",M2[i][j]);
      printf("\n");
    }*/

    //Verificação da operação escolhida e chamada das funçoes
    if(strcmp(argv[5], "o") == 0){
      if(c1==l2){
        printf("MULTIPLICAÇÃO ORIGINAL: \n");
        tempo = omp_get_wtime();
        MulM1M2(M1,M2,MR,l1,c2);   
        tempo = omp_get_wtime() - tempo;
        printf("%f\n", tempo);
        
        //Matriz Resultante
        /*printf("\n");
        for(int i=0;i<l2;i++){
          for(int j=0;j<c2;j++)
              printf("%d ",MR[i][j]);
            printf("\n");
        }*/
      }
      else{
        printf("Não é possivel realizar a multiplicação!\n");
        exit(-1);
      }
    }
    else{
      printf("Opção inválida!\n");
      exit(-1);
    }
    }
  return 0;
}