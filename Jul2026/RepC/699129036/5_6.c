/*
 * Propósito: Implementar um programa produtor-consumidor no qual alguns dos
 * threads são produtores e outros são consumidores. Os produtores leem texto de
 * uma coleção de arquivos, um por produtor. Eles inserem linhas de texto em uma
 * única fila compartilhada. Os consumidores pegam as linhas de texto e as
 * tokenizam - ou seja, identificam sequências de caracteres separadas por
 * espaços do restante da linha. Quando um consumidor encontra um token, ele
 * escreve no stdout.
 *
 * Requisitos:
 * 1. Deve haver pelo menos um produtor e um consumidor.
 * 2. Existem duas seções críticas:
 *  - Uma é protegida por uma diretiva atomic em prd_count.
 *  - A outra é protegida por diretivas críticas: uma em Enqueue e outra em
 * Dequeue.
 *
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const int MAX_FILES = 50;
const int MAX_CHAR = 100;

struct list_node_s {
    char* data;
    struct list_node_s* next;
};

void como_usar(char* prog_name);
void prd_count(int prod_count, int cons_count, FILE* files[], int file_count);
void get_arquivos(FILE* files[], int* file_count_p);
void ler_arquivo(FILE* file, struct list_node_s** queue_head,
                 struct list_node_s** queue_tail, int my_rank);
void enfileirar(char* line, struct list_node_s** queue_head,
                struct list_node_s** queue_tail);
struct list_node_s* Dequeue(struct list_node_s** queue_head,
                            struct list_node_s** queue_tail, int my_rank);
void token(char* data, int my_rank);
void printar_fila(int my_rank, struct list_node_s* queue_head);

int main(int argc, char* argv[]) {
    int prod_count, cons_count;
    FILE* files[MAX_FILES];
    int file_count;

    if (argc != 3) como_usar(argv[0]);
    prod_count = strtol(argv[1], NULL, 10);
    cons_count = strtol(argv[2], NULL, 10);

    get_arquivos(files, &file_count);

    prd_count(prod_count, cons_count, files, file_count);

    return 0;
}

void como_usar(char* prog_name) {
    fprintf(stderr, "usage: %s <producer count> <consumer count>\n", prog_name);
    exit(0);
}

void get_arquivos(FILE* files[], int* file_count_p) {
    int i = 0;
    char filename[MAX_CHAR];

    while (scanf("%s", filename) != -1) {
        files[i] = fopen(filename, "r");
        if (files[i] == NULL) {
            fprintf(stderr, "Can't open %s\n", filename);
            fprintf(stderr, "Quitting . . . \n");
            exit(-1);
        }
        i++;
    }
    *file_count_p = i;
}

void prd_count(int prod_count, int cons_count, FILE* files[], int file_count) {
    int thread_count = prod_count + cons_count;
    struct list_node_s* queue_head = NULL;
    struct list_node_s* queue_tail = NULL;
    int prod_done_count = 0;

#pragma omp parallel num_threads(thread_count) default(none)                   shared(file_count, queue_head, queue_tail, files, prod_count, cons_count,  prod_done_count)
    {
        int my_rank = omp_get_thread_num(), f;
        if (my_rank < prod_count) {
            for (f = my_rank; f < file_count; f += prod_count) {
                ler_arquivo(files[f], &queue_head, &queue_tail, my_rank);
            }
#pragma omp atomic
            prod_done_count++;
        } else {
            struct list_node_s* tmp_node;

            while (prod_done_count < prod_count) {
                tmp_node = Dequeue(&queue_head, &queue_tail, my_rank);
                if (tmp_node != NULL) {
                    token(tmp_node->data, my_rank);
                    free(tmp_node);
                }
            }
            while (queue_head != NULL) {
                tmp_node = Dequeue(&queue_head, &queue_tail, my_rank);
                if (tmp_node != NULL) {
                    token(tmp_node->data, my_rank);
                    free(tmp_node);
                }
            }
        }
    }
}

void ler_arquivo(FILE* file, struct list_node_s** queue_head,
                 struct list_node_s** queue_tail, int my_rank) {
    char* line = malloc(MAX_CHAR * sizeof(char));
    while (fgets(line, MAX_CHAR, file) != NULL) {
        printf("Th %d > read line: %s", my_rank, line);
        enfileirar(line, queue_head, queue_tail);
        line = malloc(MAX_CHAR * sizeof(char));
    }

    fclose(file);
}

void enfileirar(char* line, struct list_node_s** queue_head,
                struct list_node_s** queue_tail) {
    struct list_node_s* tmp_node = NULL;

    tmp_node = malloc(sizeof(struct list_node_s));
    tmp_node->data = line;
    tmp_node->next = NULL;

#pragma omp critical
    if (*queue_tail == NULL) {
        *queue_head = tmp_node;
        *queue_tail = tmp_node;
    } else {
        (*queue_tail)->next = tmp_node;
        *queue_tail = tmp_node;
    }
}

struct list_node_s* Dequeue(struct list_node_s** queue_head,
                            struct list_node_s** queue_tail, int my_rank) {
    struct list_node_s* tmp_node = NULL;

    if (*queue_head == NULL) return NULL;

#pragma omp critical
    {
        if (*queue_head == *queue_tail) *queue_tail = (*queue_tail)->next;

        tmp_node = *queue_head;
        *queue_head = (*queue_head)->next;
    }

    return tmp_node;
}

void token(char* data, int my_rank) {
    char *my_token, *word;

    my_token = strtok_r(data, " \t\n", &word);
    while (my_token != NULL) {
        printf("Th %d token: %s\n", my_rank, my_token);
        my_token = strtok_r(NULL, " \t\n", &word);
    }
}

void printar_fila(int my_rank, struct list_node_s* queue_head) {
    struct list_node_s* curr_p = queue_head;

    printf("Th %d > queue = \n", my_rank);
#pragma omp critical
    while (curr_p != NULL) {
        printf("%s", curr_p->data);
        curr_p = curr_p->next;
    }
    printf("\n");
}