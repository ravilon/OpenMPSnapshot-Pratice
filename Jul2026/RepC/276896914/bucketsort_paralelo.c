/*

-----  Projeto Final -----

Versão paralela do algoritmo de ordenação Bucket Sort.
Código retirado de: <https://www.programiz.com/dsa/bucket-sort>.  

Nome: Mateus Grota Nishimura Ferro.

Projeto final da ACIEPE(Atividade Curricular de Integração entre Ensino, Pesquisa e Extensão) - "Programação paralela:
das threads aos FPGAs - uma introdução ao tema(1002004)", ofertada pela UFSCar - São Carlos.

---------------------------
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define NARRAY 10000  // Número de elementos do vetor
#define NBUCKET 100  // Quantidade de baldes
#define INTERVAL 100  // Intervalo de elementos para cada balde

struct Node {
  int data;
  struct Node *next;
};

void BucketSort(int arr[]);
struct Node *InsertionSort(struct Node *list);
void print(int arr[]);
void printBuckets(struct Node *list);
int getBucketIndex(int value);

void BucketSort(int arr[]) {
  int i, j;
  struct Node **buckets;

  // Criação dos baldes
  buckets = (struct Node **)malloc(sizeof(struct Node *) * NBUCKET);

  // Começo da região paralela
  #pragma omp parallel num_threads(omp_get_num_threads())
  {
    // Inicializa os baldes vazios
    #pragma omp for schedule(static)
    for (i = 0; i < NBUCKET; ++i) {
      buckets[i] = NULL;
    }

    // Preenche os baldes vazios com seus respectivos elementos pertencentes ao seu intervalo.
    #pragma omp for schedule(static)
    for (i = 0; i < NARRAY; ++i) {
        struct Node *current;
        int pos = getBucketIndex(arr[i]);
        current = (struct Node *)malloc(sizeof(struct Node));
        current->data = arr[i];
        current->next = buckets[pos];
        buckets[pos] = current;
    }

    // Imprime os elementos de cada balde
    #pragma omp single
    for (i = 0; i < NBUCKET; i++) {
      printf("Bucket[%d]: ", i);
      printBuckets(buckets[i]);
      printf("\n");
    }

    // Ordena os elementos de cada balde
    #pragma omp for schedule(dynamic)
    for (i = 0; i < NBUCKET; ++i) {
      buckets[i] = InsertionSort(buckets[i]);
    }

    printf("-------------\n");
    printf("Bucktets after sorting\n");
    #pragma omp single
    for (i = 0; i < NBUCKET; i++) {
      printf("Bucket[%d]: ", i);
      printBuckets(buckets[i]);
      printf("\n");
    }

    // Concantenação dos elementos no vetor
    #pragma omp single
    for (j = 0, i = 0; i < NBUCKET; ++i) {
      struct Node *node;
      node = buckets[i];
      while (node) {
        arr[j++] = node->data;
        node = node->next;
      }
  }

  }
  return;
}

// Ordenação de cada balde feite pelo Insertion Sort
struct Node *InsertionSort(struct Node *list) {
  struct Node *k, *nodeList;
  if (list == 0 || list->next == 0) {
    return list;
  }

  nodeList = list;
  k = list->next;
  nodeList->next = 0;
  while (k != 0) {
    struct Node *ptr;
    if (nodeList->data > k->data) {
      struct Node *tmp;
      tmp = k;
      k = k->next;
      tmp->next = nodeList;
      nodeList = tmp;
      continue;
    }

    for (ptr = nodeList; ptr->next != 0; ptr = ptr->next) {
      if (ptr->next->data > k->data)
        break;
    }

    if (ptr->next != 0) {
      struct Node *tmp;
      tmp = k;
      k = k->next;
      tmp->next = ptr->next;
      ptr->next = tmp;
      continue;
    } else {
      ptr->next = k;
      k = k->next;
      ptr->next->next = 0;
      continue;
    }
  }
  return nodeList;
}

int getBucketIndex(int value) {
  return value / INTERVAL;
}

void print(int ar[]) {
  int i;
  for (i = 0; i < NARRAY; ++i) {
    printf("%d ", ar[i]);
  }
  printf("\n");
}

// Imprime os baldes
void printBuckets(struct Node *list) {
  struct Node *cur = list;
  while (cur) {
    printf("%d ", cur->data);
    cur = cur->next;
  }
}


int main(void) {
  int array[NARRAY], i;

   srand(time(NULL));
   for(i=0; i < NARRAY; i++)
      array[i]= (int)rand() / (int)(RAND_MAX/ 10000);

  printf("Initial array: ");
  print(array);
  printf("-------------\n");

  BucketSort(array);
  printf("-------------\n");
  printf("Sorted array: ");
  print(array);
  return 0;
}