#include <stdio.h>
#include <time.h>
#include <limits.h>
#include <stdlib.h>
#include <omp.h>

int min(int a, int b)
{
return (((a) < (b)) ? (a) : (b));
}
int max(int a, int b)
{
return (((a) > (b)) ? (a) : (b));
}

void dijkstra(int **graph, int n, int startnode);

struct point
{
int mindistance;
int index;
};

#pragma omp declare reduction(min : struct point : omp_out.mindistance = omp_in.mindistance > omp_out.mindistance ? omp_out.mindistance : omp_in.mindistance,  omp_out.index = omp_in.mindistance > omp_out.mindistance ? omp_out.index : omp_in.index)                                     initializer(omp_priv = {INT_MAX, INT_MAX})

int main(int argc, char *argv[])
{
int n, u, **graph, t;
clock_t start, end;

FILE *inputFile = fopen("graphInput.txt", "r");
if (!inputFile)
{
perror("Error opening the input file");
return 1;
}

fscanf(inputFile, "%d", &n);

printf("Kanak Khandelwal 2021BCS0102\n");

graph = (int **)malloc(n * sizeof(int *));
for (int i = 0; i < n; i++)
{
graph[i] = (int *)malloc(n * sizeof(int));
}

for (int i = 0; i < n; i++)
{
for (int j = 0; j < n; j++)
{
fscanf(inputFile, "%d", &graph[i][j]);
}
}
fscanf(inputFile, "%d", &u);
fclose(inputFile);

printf("Starting Node: %d\n", u);

srand(time(NULL));
start = clock();
dijkstra(graph, n, u);
end = clock();
double tme = ((double)(end - start)) / CLOCKS_PER_SEC;
printf("\nThe algorithm took %f seconds to run.\n", tme);

return 0;
}

void dijkstra(int **graph, int n, int startnode)
{
int **cost, *distance, *predecessor, *visited;
cost = (int **)malloc(n * sizeof(int *));
for (int i = 0; i < n; i++)
{
cost[i] = (int *)malloc(n * sizeof(int));
}
distance = (int *)malloc(n * sizeof(int));
predecessor = (int *)malloc(n * sizeof(int));
visited = (int *)malloc(n * sizeof(int));
for (int i = 0; i < n; i++)
for (int j = 0; j < n; j++)
if (graph[i][j] == 0)
cost[i][j] = INT_MAX;
else
cost[i][j] = graph[i][j];
for (int i = 0; i < n; i++)
{
distance[i] = cost[startnode][i];
predecessor[i] = startnode;
visited[i] = 0;
}

distance[startnode] = 0;
visited[startnode] = 1;
int count = 1;
while (count < n - 1)
{
struct point mini = {INT_MAX, INT_MAX}, ans = {INT_MAX, INT_MAX};
int i, j;
#pragma omp parallel for private(j) firstprivate(mini) reduction(min : ans)
for (i = 0; i < n; i++)
{
if (distance[i] < mini.mindistance && !visited[i])
{
mini.mindistance = distance[i];
mini.index = i;
ans.mindistance = mini.mindistance;
ans.index = mini.index;
}
}
int nextnode = ans.index;
visited[nextnode] = 1;
mini.mindistance = ans.mindistance;
mini.index = ans.index;

#pragma omp parallel for
for (int i = 0; i < n; i++)
{
if (!visited[i])
{
if (cost[nextnode][i] != INT_MAX && mini.mindistance + cost[nextnode][i] < distance[i])
{
distance[i] = mini.mindistance + cost[nextnode][i];
predecessor[i] = nextnode;
}
}
}
count++;
}
for (int i = 0; i < n; i++)
if (i != startnode)
{
printf("\nDistance of node %d = %d", i, distance[i]);
printf("\nPath = %d", i);

int j = i;
do
{
j = predecessor[j];
printf("<-%d", j);
} while (j != startnode);
}
}