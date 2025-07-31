/*
Versão otimizada do algoritmo base do Selection Sort utilizando 
a diretiva do OpenMP #pragma omp simd com a cláusula reduction.

Compilação: gcc -o selectionReduction -fopenmp selectionReduction.c
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

/* Estrutura utilizada para armazenar o valor e o indice do menor elemento do vetor */
struct Compara{
int valor;
int indice;
};

/* Cria uma copia privada para cada execução para armazenar o menor valor de cada vetor em execução */
#pragma omp declare reduction(min : struct Compara : omp_out = omp_in.valor > omp_out.valor ? omp_out : omp_in)
void selectionsort(int *vet, int tam);
void exibevetor(int *vet, int tam);


void selectionsort(int *vet, int tam){   
int i,j,aux;
struct Compara menor;
/* diretiva omp simd aplicada ao laço indica que multiplas iteracoes do loop podem ser executadas concorrentemente usando instrucoes SIMD */
#pragma omp simd reduction(min:menor)  //vetorizacao com reducao
for (i = 0; i < (tam - 1); ++i){ 
menor.valor = vet[i];
menor.indice = i;  //armazena indice do menor elemento
for (j = i + 1; j < tam; ++j){
if (vet[j] < menor.valor){ // busca pelo elemento de menor valor 
menor.valor = vet[j];
menor.indice = j; // salva o novo indice como menor 
}
}
/* troca e coloca o menor elemento para frente */
aux = vet[i];
vet[i] = menor.valor;
vet[menor.indice] = aux;
}
}


/* Função utilizada para exibição do vetor */
void exibevetor(int *vet, int tam){
int i;
for (i = 0; i < tam; ++i){
printf("%d ", vet[i]);
}
printf("\n");
}

int main(){
int *vet, i, tam;
clock_t t, end;
double cpu_time_used;

printf("Digite o tamanho do vetor:\n");
scanf("%d",&tam);
vet = (int *)malloc(sizeof(int)*tam);
if(vet == NULL){
exit(1);
}

for(i = 0; i < tam; ++i){
vet[i] = rand() % 100; //gera o vetor com valores pseudo-aleatorios
}

t = clock();
exibevetor(vet,tam);
selectionsort(vet, tam);
t = clock()-t;
exibevetor(vet,tam);
cpu_time_used = ((double)t)/CLOCKS_PER_SEC;
printf("\nTempo de execução: %f\n", cpu_time_used);
free(vet);

return 0;
}