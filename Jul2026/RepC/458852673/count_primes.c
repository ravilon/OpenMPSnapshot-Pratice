#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int l_prime(int a, int b);
int p_prime_v1(int a, int b);
int p_prime_v2(int a, int b);
int p_prime_v3(int a, int b);
int p_prime_v4(int a, int b);

int main(int argc, char **argv){
if(!argv[1] || !argv[2] || !argv[3]){
printf("Enter range, parallel? s/n when calling the command (e.g ./file.out a b s/n)\n");
exit(1);
}

int a = atoi(argv[1]), b = atoi(argv[2]) * 1e3;
int total = 0;

double start = omp_get_wtime();

total = (*argv[3] == 110) ? l_prime(a, b) : p_prime_v4(a, b);

double end = omp_get_wtime();

printf("total:%d\n%f\n", total, end - start);

// FILE *file = fopen("result.csv", "a+");
// fprintf(file, "%s;%f;%d;\n", "v2", end - start, b);
// fclose(file);

return 0;
}

int p_prime_v3(int a, int b){
int temp_total = 0;
int total = 0;
if(a == 1 || a == 0) a = 2;

#pragma omp parallel shared(a, b) reduction(+:total)
{
#pragma omp for
for(int atual = a; atual <= b; atual++){
int ePrimo = 1;

for(int divisor = 2; divisor < atual; divisor++){
if(atual % divisor == 0) ePrimo = 0;
}

if(ePrimo == 1) total++;
}
}

return total;
}

int p_prime_v4(int a, int b){
int total = 0;
if(a == 1 || a == 0) a = 2;

#pragma omp parallel for reduction(+:total) shared(a, b) 
for(int atual = a; atual <= b; atual++){
int ePrimo = 1;

for(int divisor = 2; divisor < atual && ePrimo == 1; divisor++){
if(atual % divisor == 0) ePrimo = 0;
}

if(ePrimo) total++;
}

return total;
}

int p_prime_v2(int a, int b){
int temp_total = 0;
int total = 0;
if(a == 1 || a == 0) a = 2;

#pragma omp parallel firstprivate(temp_total) shared(a, b, total)
{
#pragma omp for nowait
for(int atual = a; atual <= b; atual++){
int ePrimo = 1;

for(int divisor = 2; divisor < atual && ePrimo == 1; divisor++){
if(atual % divisor == 0) ePrimo = 0;
}

if(ePrimo) temp_total++;
}

#pragma omp atomic
total += temp_total;
}

return total;
}

int p_prime_v1(int a, int b){
int temp_total = 0;
int total = 0;
if(a == 1 || a == 0) a = 2;

#pragma omp parallel firstprivate(temp_total) shared(a, b, total)
{
#pragma omp for nowait
for(int atual = a; atual <= b; atual++){
int ePrimo = 1;

for(int divisor = 2; divisor < atual; divisor++){
if(!ePrimo) continue;
if(atual % divisor == 0) ePrimo = 0;
}

if(ePrimo) temp_total++;
}

#pragma omp atomic
total += temp_total;
}

return total;
}

int l_prime(int a, int b){
int total = 0;

for(int atual = a; atual <= b; atual++){
int ePrimo = 1;

for(int divisor = 2; divisor < atual && atual != 0 && atual != 1; divisor++){
if(!ePrimo) continue;
if(atual % divisor == 0) ePrimo = 0;
}

if(ePrimo) total++;
}

return total;
}