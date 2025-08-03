#include <stdio.h>
#include <stdlib.h>

#include "structs.h"
#include "utils.h"

/*--------------------------------
Funçao: readFile
Objectivo: Reservar a memoria necessaria na estrutura de dados LcsMAtrix para resolver o problema
Variaveis/Memoria:
lcs -> estrutura de dados que guarda toda a memoria necessaria
lcs->mtx -> matriz principal
lcs->seqColumn, lcs->seqLine -> guardam as duas strings do ficheiro. Lines é a primeira a ser lida e estará ligada
ás linhas da matriz. O mesmo para a segunda string e seqColumn, mas estará ligada as colunas da matriz
lcs->lines, lcs->cols -> guardam as dimensoes de cada string. Util para a seguir definir o tamanho da matriz
---------------------------------*/

LcsMatrix* readFile(FILE *fp) {

int i;
LcsMatrix *lcs = (LcsMatrix *)calloc(1, sizeof(LcsMatrix));
checkNullPointer(lcs);

// ler as dimensões (primeira linha do ficheiro)
if (fscanf(fp, "%d %d", &(lcs->lines), &(lcs->cols)) != 2) {
puts("Error reading file");
exit(-1);
}

// alloc matrix space
lcs->mtx = (int **)calloc(lcs->lines+1, sizeof(int **));
checkNullPointer(lcs->mtx);

#pragma omp parallel for schedule(dynamic) private(i)	//todas as colunas da matriz sao completamente independentes umas das outras. Assim, pode-se paralelizar facilmente
for(i=0; i<lcs->lines+1; i++) {
lcs->mtx[i] = (int *)calloc(lcs->cols+1, sizeof(int *));
checkNullPointer(lcs->mtx[i]);
}

// alloc sequence space
// +2 por causa do espaço inicial(para facilidade na programação) e do \0 no fim
lcs->seqLine = (char *)calloc(lcs->lines+2, sizeof(char));
checkNullPointer(lcs->seqLine);
lcs->seqColumn = (char *)calloc(lcs->cols+2, sizeof(char));
checkNullPointer(lcs->seqColumn);

// ler as strings e guardá-las
if (fscanf(fp, "%s", &lcs->seqLine[1]) != 1) {
puts("Error reading file");
exit(-1);
}
lcs->seqLine[0] = ' ';
lcs->seqLine[lcs->lines+1]='\0';
if (fscanf(fp, "%s", &lcs->seqColumn[1]) != 1) {
puts("Error reading file");
exit(-1);
}
lcs->seqColumn[0] = ' ';
lcs->seqColumn[lcs->cols+1]='\0';

return lcs;
}
