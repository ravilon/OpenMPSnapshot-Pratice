#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#define n 1000 //number of nodes

void showDistances(int** dist);

int main(int argc, char** argv) 
{

int i, j, k;
int** dist; //array with the distances between nodes

//Initiate the necessary memory with malloc()
dist = (int**)malloc(n*sizeof(int*));
for(i=0; i<n; i++)
dist[i] = (int*)malloc(n*sizeof(int));

time_t start, end;
//use current time
time(&start);
//to generate "random" numbers with rand()
srand(42);

//Initiate the dist with random values from 0-99
for(i=0; i<n; i++)
for(j=0; j<n; j++)
if(i==j)
dist[i][j] = 0;
else
dist[i][j] = rand()%100;

//Print initial distances
showDistances(dist);	

time(&start);
//Calculate minimum distance paths
//Using omp parallel for, it partitions the loop into the threads (as many as the CPUs) and runs the algorithm
#pragma omp parallel for private(i,j,k) shared(dist)
for(k=0; k<n; k++) 
for(i=0; i<n; i++)
for(j=0; j<n; j++)
if ((dist[i][k] * dist[k][j] != 0) && (i != j))
if(dist[i][j] > dist[i][k] + dist[k][j] || dist[i][j] == 0)
dist[i][j] = dist[i][k] + dist[k][j];

time(&end);
//print the final distances
showDistances(dist);

printf("Total Elapsed Time %f sec\n", difftime(end, start));	
free(dist);
return 0;
}


//Print distance function
void showDistances(int** dist) 
{

int i, j;
printf("     ");
for(i=0; i<n; ++i)
printf("N%d   ", i);
printf("\n");
for(i=0; i<n; ++i) {
printf("N%d", i);
for(j=0; j<n; ++j)
printf("%5d", dist[i][j]);
printf("\n");
}
printf("\n");
}	




