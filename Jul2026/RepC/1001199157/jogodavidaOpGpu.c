#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

#define ind2d(i,j) ((i)*(tam+2)+(j))
#define POWMIN 3
#define POWMAX 10

double wall_time(void) {
struct timeval tv;
gettimeofday(&tv, NULL);
return (tv.tv_sec + tv.tv_usec / 1000000.0);
}

void DumpTabul(int *tabul, int tam, int first, int last, char* msg) {
int i, ij;
printf("%s; Dump [%d:%d] de tabuleiro %d x %d\n", msg, first, last, tam, tam);
for (i=first; i<=last; i++) printf("="); printf("=\n");
for (i=ind2d(first,0); i<=ind2d(last,0); i+=ind2d(1,0)) {
for (ij=i+first; ij<=i+last; ij++)
printf("%c", tabul[ij] ? 'X' : '.');
printf("\n");
}
for (i=first; i<=last; i++) printf("="); printf("=\n");
}

void InitTabul(int* tabulIn, int* tabulOut, int tam) {
int ij;
for (ij = 0; ij < (tam + 2) * (tam + 2); ij++) {
tabulIn[ij] = 0;
tabulOut[ij] = 0;
}
// Glider
tabulIn[ind2d(1,2)] = 1; tabulIn[ind2d(2,3)] = 1;
tabulIn[ind2d(3,1)] = 1; tabulIn[ind2d(3,2)] = 1;
tabulIn[ind2d(3,3)] = 1;
}

int Correto(int* tabul, int tam) {
int ij, cnt = 0;
for (ij = 0; ij < (tam + 2) * (tam + 2); ij++)
cnt += tabul[ij];
return (cnt == 5 &&
tabul[ind2d(tam - 2, tam - 1)] &&
tabul[ind2d(tam - 1, tam)] &&
tabul[ind2d(tam, tam - 2)] &&
tabul[ind2d(tam, tam - 1)] &&
tabul[ind2d(tam, tam)]);
}

void UmaVidaGPU(int* tabulIn, int* tabulOut, int tam) {
#pragma omp target teams distribute parallel for collapse(2)  map(to: tabulIn[0:(tam+2)*(tam+2)])  map(from: tabulOut[0:(tam+2)*(tam+2)])
for (int i = 1; i <= tam; i++) {
for (int j = 1; j <= tam; j++) {
int vizviv = tabulIn[ind2d(i-1,j-1)] + tabulIn[ind2d(i-1,j)] +
tabulIn[ind2d(i-1,j+1)] + tabulIn[ind2d(i,j-1)] +
tabulIn[ind2d(i,j+1)] + tabulIn[ind2d(i+1,j-1)] +
tabulIn[ind2d(i+1,j)] + tabulIn[ind2d(i+1,j+1)];

int atual = tabulIn[ind2d(i,j)];
if (atual && vizviv < 2)
tabulOut[ind2d(i,j)] = 0;
else if (atual && vizviv > 3)
tabulOut[ind2d(i,j)] = 0;
else if (!atual && vizviv == 3)
tabulOut[ind2d(i,j)] = 1;
else
tabulOut[ind2d(i,j)] = atual;
}
}
}

int main(void) {
int pow;
int i, tam, *tabulIn, *tabulOut;
double t0, t1, t2, t3;

for (pow = POWMIN; pow <= POWMAX; pow++) {
tam = 1 << pow;
t0 = wall_time();

tabulIn  = (int *) malloc((tam+2)*(tam+2)*sizeof(int));
tabulOut = (int *) malloc((tam+2)*(tam+2)*sizeof(int));
InitTabul(tabulIn, tabulOut, tam);

//transferência inicial dos dados para GPU
#pragma omp target enter data map(to: tabulIn[0:(tam+2)*(tam+2)],  tabulOut[0:(tam+2)*(tam+2)])

t1 = wall_time();

for (i = 0; i < 2*(tam-3); i++) {
UmaVidaGPU(tabulIn, tabulOut, tam);
UmaVidaGPU(tabulOut, tabulIn, tam);
}

//verificar resultado
#pragma omp target update from(tabulIn[0:(tam+2)*(tam+2)])
t2 = wall_time();

if (Correto(tabulIn, tam))
printf("**RESULTADO CORRETO**\n");
else
printf("**RESULTADO ERRADO**\n");

t3 = wall_time();

printf("tam=%d; tempos: init=%7.7f, comp=%7.7f, fim=%7.7f, tot=%7.7f\n",
tam, t1-t0, t2-t1, t3-t2, t3-t0);

// Liberação da memória na GPU
#pragma omp target exit data map(delete: tabulIn[0:(tam+2)*(tam+2)],  tabulOut[0:(tam+2)*(tam+2)])

free(tabulIn);
free(tabulOut);
}

return 0;
}
