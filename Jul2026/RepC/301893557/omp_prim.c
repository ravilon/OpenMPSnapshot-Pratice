#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdio.h>
#include <omp.h>
#include "omp_prim.h"

/*
Obtem a aresta de menor custo dentro da lista de arestas candidatas passadas.

Recebe:

- edges: lista de arestas candidatas, cada aresta foi encontrada por um thread em sua area de procura

Retorna:

- aresta de menor custo encontrada na lista ou NULL caso a lista seja vazia
*/
Edge *get_minimum_cost_edge(Edge *edges)
{
if (!edges)
return NULL;

if (!edges->next)
return edges;

Edge *best_edge = edges;
for (Edge *ptr = edges->next; ptr != NULL; ptr = ptr->next)
if (ptr->cost < best_edge->cost)
best_edge = ptr;

return best_edge;
}

/*
Versao paralela em OpenMP do algoritmo de Prim.

Recebe:

- cost: matriz de adjacencias contendo os custos de cada aresta;
- rows: total de linhas;
- columns: total de colunas;
- nthreads: numero de threads de execucao a utilizar;
- ntrials: numero de repeticoes do cenario, para calculo das medias de tempo de execucao;
- line: linha da tabela de resultados a ser preenchida com o tempo de execucao da funcao

Retorna:

- custo minimo da MST encontrada pelo algoritmo
*/
long omp_prim_minimum_spanning_tree(int **cost, int rows, int columns, int nthreads, int ntrials, Table *line)
{
double partial_time = 0.0; //inicializa tempo parcial de execucao
long minimum_cost;         //define custo da MST

//Itera ntrials vezes, agregando o tempo parcial de execucao, para depois calcular sua media
for (int i = 0; i < ntrials; i++)
{
//marca tempo de inicio da presente execucao usando OMP
partial_time -= omp_get_wtime();
//usa o numero de threads passado para execucao
omp_set_num_threads(nthreads);
//inicializa lista vazia de nos ja inclusos na MST parcial
int *vertices_in_mst = (int *)malloc(rows * sizeof(int));
memset(vertices_in_mst, 0, rows * sizeof(int));
//insere primeiro no do grafo na lista de nos ja inclusos na MST parcial
vertices_in_mst[0] = 1;

//custo atual é zero
minimum_cost = 0;
//qtd. de arestas inclusas na solucao é zero inicialmente
int edge_count = 0;
//enquanto o total de arestas for menor que o numero de nos-1, itera
//buscando a proxima aresta a ser inserida na MST
while (edge_count < rows - 1)
{
//inicializa lista auxiliar de arestas candidatas para insercao
//usada na atual iteracao
Edge *edges = NULL;

//indica inicio de bloco paralelo
#pragma omp parallel shared(edges, cost, rows, columns, edge_count, vertices_in_mst)
{
//inicializa dados usados para marcacao da aresta de menor custo encontrada
//pela thread
int min = INT_MAX, a = -1, b = -1;

//itera de forma paralela sobre todas as linhas
#pragma omp parallel for
for (int i = 0; i < rows; i++)
{
//itera de forma paralela sobre todas as colunas
#pragma omp parallel for
for (int j = 0; j < columns; j++)
{   
//se a aresta atual tiver custo menor que a aresta de custo minimo atual,
//e se for uma aresta passivel de insercao na MST parcial...
if (cost[i][j] < min && is_valid_edge(i, j, vertices_in_mst))
{
//...marca a aresta como a melhor encontrada
min = cost[i][j];
a = i;
b = j;
}
}
}

//se alguma aresta candidata de menor custo foi encontrada no espaco
//de procura da presente thread, insere tal aresta na lista compartilhada
//de arestas candidatas
if (a != -1 && b != -1 && min != INT_MAX)
{
Edge *edge = create_edge_node(a, b, min);
//usa a diretiva critical para garantir que a operacao de
//insercao de uma aresta na lista de candidatas será executada
//em uma thread por vez
#pragma omp critical
edges = insert_node(edge, edges);
}
}

//na thread principal, verifica se a lista de arestas candidatas possui algum elemento
if (edges != NULL)
{
//caso positivo, busca dentro da lista a aresta com o menor custo dentre
//as existentes
Edge *best_edge = get_minimum_cost_edge(edges);

//exibe a aresta selecionada para integrar a solucao parcial da MST
printf("Selected edge %d:(%d, %d), cost: %d\n", edge_count, best_edge->a, best_edge->b, best_edge->cost);
//atualiza o custo minimo da MST parcial, assim como
//a contagem de arestas inclusas e de os nós já inclusos na MST
minimum_cost = minimum_cost + best_edge->cost;
vertices_in_mst[best_edge->b] = vertices_in_mst[best_edge->a] = 1;
edge_count++;
//libera lista de arestas candidatas
free_edge_list(edges);
}
}
//exibe o custo da minimum spanning tree encontrada pelo algoritmo
printf("MST cost: %ld\n", minimum_cost);

//libera lista de vertices ja inclusos na MST
free(vertices_in_mst);

//marca o fim da execucao desta iteracao no tempo parcial usando OMP
partial_time += omp_get_wtime();
}
//coloca a media do tempo parcial de execucao na linha da tabela CSV passada
line->execution_time = partial_time / ntrials;
//retorna o custo da MST encontrada pelo algoritmo
return minimum_cost;
}