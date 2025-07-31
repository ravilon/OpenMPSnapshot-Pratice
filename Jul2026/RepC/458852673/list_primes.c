#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

typedef struct lista {
int info;
struct lista* prox;
} TLista;

TLista* insere_inicio(TLista *lista, int info);
TLista* insere_fim(TLista *lista, int info);
TLista* l_prime(int a, int b);
TLista* p_prime(int a, int b);

int main(int argc, char **argv){
omp_set_num_threads(atoi(argv[4]));

if(!argv[1] || !argv[2] || !argv[3]){
printf("Enter range, parallel? s/n when calling the command (e.g ./file.out a b s/n)\n");
exit(1);
}

int a = atoi(argv[1]), b = atoi(argv[2]) * 1e3;

double start = omp_get_wtime();

TLista *list = (*argv[3] == 110) ? l_prime(a, b) : p_prime(a, b);

double end = omp_get_wtime();

while(list){
printf("%d", list->info);
list = list->prox;
if(list) printf(", ");
}

printf("\n");
printf("%f\n", end - start);

return 0;
}

TLista* insere_inicio(TLista *lista, int info){
TLista* novo = (TLista*)malloc(sizeof(TLista));
novo->info = info;
novo->prox = lista;
return novo;
}

TLista* concat(TLista *from, TLista *to){
if(!to) return from;

TLista* temp = to;
while(temp->prox) temp = temp->prox;

temp->prox = from;

return temp;
}

TLista* p_prime(int a, int b){
TLista *list = NULL, *end_list = NULL;
if(a == 1 || a == 0) a = 2;

#pragma omp parallel shared(a, b, list)
{
TLista *temp_list = NULL;

#pragma omp for nowait
for(int atual = a; atual <= b; atual++){
int ePrimo = 1;

for(int divisor = 2; divisor < atual && ePrimo != 0; divisor++){
if(atual % divisor == 0) ePrimo = 0;
}

if(ePrimo == 1) insere_inicio(temp_list, atual);
}

#pragma omp critical
{
if(!list){
end_list = concat(temp_list, end_list);
list = end_list;
} else {
end_list = concat(temp_list, end_list);
}
}

}

return list;
}

TLista* l_prime(int a, int b){
TLista *list = NULL;
if(a == 1 || a == 0) a = 2;

for(int atual = a; atual <= b; atual++){
int ePrimo = 1;

for(int divisor = 2; divisor < atual && ePrimo != 0; divisor++){
if(atual % divisor == 0) ePrimo = 0;
}

if(ePrimo == 1) insere_inicio(list, atual);

}

return list;
}