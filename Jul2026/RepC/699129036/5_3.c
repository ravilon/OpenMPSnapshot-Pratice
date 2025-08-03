#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "timer.h"

/*
* (a) i, j e count devem ser privados; temp, a e n devem ser compartilhados.
*
* (b) Não. As únicas variáveis (além das variáveis privadas do loop) que são
* escritas são count e temp. Count é privado e cada thread escreve apenas seus
* próprios elementos de temp.
*
* (c) Poderíamos fazer com que cada thread identifique um subbloco de a e temp
* e depois chame memcpy em seu subbloco. No entanto, isso pode ser arriscado
* sem conhecer os detalhes da implementação do memcpy. Uma alternativa mais
* segura é usar um loop for para copiar temp para a, e então paralelizar isso
* com uma diretiva for.
*
* (d) Veja abaixo.
*
* (e) Em um dos nossos sistemas, a ordenação paralela por contagem atinge
* speedup linear com 8 threads e n = 10.000. No entanto, a ordenação por
* contagem é O(n^2). Portanto, mesmo com 8 threads, ela ainda é mais lenta que
* o qsort.
*
*/

void como_usar(char prog_name[]);
void get_argumentos(char* argv[], int* thread_count_p, int* n_p);
void gerar_dados(int a[], int n);
void csort_s(int a[], int n);
void csort_p(int a[], int n, int thread_count);
void lib_qsort(int a[], int n);
int comp(const void* a, const void* b);
void print_dados(int a[], int n, char msg[]);
int chsort(int a[], int n);

int main(int argc, char* argv[]) {
int n, thread_count;
int *a, *copy;
double start, stop;

if (argc != 3) como_usar(argv[0]);
get_argumentos(argv, &thread_count, &n);

a = malloc(n * sizeof(int));
gerar_dados(a, n);

copy = malloc(n * sizeof(int));

memcpy(copy, a, n * sizeof(int));
csort_s(copy, n);
if (!chsort(copy, n)) printf("Serial sort failed\n");
printf("Serial run time: %e\n\n", stop - start);

memcpy(copy, a, n * sizeof(int));
csort_p(copy, n, thread_count);
if (!chsort(copy, n)) printf("Parallel sort failed\n");
printf("Parallel run time: %e\n\n", stop - start);

memcpy(copy, a, n * sizeof(int));
lib_qsort(copy, n);
if (!chsort(copy, n)) printf("Library sort failed\n");
printf("qsort run time: %e\n", stop - start);

free(a);
free(copy);

return 0;
}

void como_usar(char prog_name[]) {
fprintf(stderr, "usage: %s <thread_count> <n>\n", prog_name);
exit(0);
}

void get_argumentos(char* argv[], int* thread_count_p, int* n_p) {
*thread_count_p = strtol(argv[1], NULL, 10);
*n_p = strtol(argv[2], NULL, 10);
}

void gerar_dados(int a[], int n) {
int i;

for (i = 0; i < n; i++) a[i] = random() % n + 1;  // (double) RAND_MAX;
}

void csort_s(int a[], int n) {
int i, j, count;
int* temp = malloc(n * sizeof(int));

for (i = 0; i < n; i++) {
count = 0;
for (j = 0; j < n; j++)
if (a[j] < a[i])
count++;
else if (a[j] == a[i] && j < i)
count++;
temp[count] = a[i];
}

memcpy(a, temp, n * sizeof(int));
free(temp);
}

void csort_p(int a[], int n, int thread_count) {
int i, j, count;
int* temp = malloc(n * sizeof(int));

#pragma omp parallel num_threads(thread_count) default(none) private(  i, j, count) shared(n, a, temp)
{
#pragma omp for
for (i = 0; i < n; i++) {
count = 0;
for (j = 0; j < n; j++)
if (a[j] < a[i])
count++;
else if (a[j] == a[i] && j < i)
count++;
temp[count] = a[i];
}

#pragma omp for
for (i = 0; i < n; i++) a[i] = temp[i];
}

free(temp);
}

void lib_qsort(int a[], int n) { qsort(a, n, sizeof(int), comp); }

int comp(const void* a, const void* b) {
const int* int_a = (const int*)a;
const int* int_b = (const int*)b;

return (*int_a - *int_b);
}

void print_dados(int a[], int n, char msg[]) {
int i;

printf("%s = ", msg);
for (i = 0; i < n; i++) printf("%d ", a[i]);
printf("\n");
}

int chsort(int a[], int n) {
int i;

for (i = 1; i < n; i++)
if (a[i - 1] > a[i]) return 0;
return 1;
}