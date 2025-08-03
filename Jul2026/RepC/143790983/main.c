//gcc-7 -fopenmp -o main main.c
//./main

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N   20     //matrix size
#define NUM_THREADS 4 // count of threads
#define LOOP_COUNT 10 // Repeat (b) at least 10 times and observe the changes in the matrix.


void printMatrix (int matrix[N][N]);
void printMatrix2 (int matrix[N+2][N+2]);
int checkNeighbours(int matrix[N+2][N+2],int i,int j);

int main() {
//Set thread size
omp_set_num_threads(NUM_THREADS);

int input[N][N];
int x[N+2][N+2];//Temp matrix for operations
int result[N+2][N+2];
//Assign random values to each matrix element (0-dead, 1-alive)
for (int i=0; i<N; i++) {
for (int j=0; j<N; j++) {
input[i][j]= rand() % 2;
}
}

//Show input matrix
printMatrix(input);

//Init result matrix
for (int i=0; i<N+2; i++) {
for (int j=0; j<=N+2; j++) {
result[i][j] = 0;
}
}

//Pad borders with zero and copy input matrix to temp matrix
for (int i=0; i<=N+1; i++) {
for (int j=0; j<=N+1; j++) {
if(i==0 || j==0 || i==N+1 || j==N+1)
x[i][j] = 0;
else
x[i][j]= input[i-1][j-1];
}
}

int n=N;
for(int i=0; i<LOOP_COUNT; i++)
{
printf("Step %d\n \n",i+1);
#pragma omp parallel for shared (n)
for (int j=1; j < n; j++)
{
for (int i=1; i < n; i++)
{
int temp= checkNeighbours(x,i,j);
result[i][j]=temp;
}
}
//Copy output matrix for next iteration
for (int i=0; i<N+2; i++) {
for (int j=0; j<=N+2; j++) {
x[i][j] = result[i][j];
}
}
printMatrix2(result);
}
}

//. Change the state of each element using its current state and the states of its 8 neighbours:
//- if there are 5 or more 1s, change its state to alive
//- otherwise change its state to dead
int checkNeighbours(int matrix[N+2][N+2],int i,int j)
{
int aliveCount = 0, deadCount = 0;

matrix[i-1][j-1] == 0 ? deadCount++ : aliveCount++;
matrix[i-1][j] == 0 ? deadCount++ : aliveCount++;
matrix[i-1][j+1] == 0 ? deadCount++ : aliveCount++;
matrix[i][j-1] == 0 ? deadCount++ : aliveCount++;
matrix[i][j+1] == 0 ? deadCount++ : aliveCount++;
matrix[i+1][j-1] == 0 ? deadCount++ : aliveCount++;
matrix[i+1][j] == 0 ? deadCount++ : aliveCount++;
matrix[i+1][j+1] == 0 ? deadCount++ : aliveCount++;

if(aliveCount>= 5)
return 1;
else
return 0;
}

void printMatrix (int matrix[N][N])
{
int i,j;
for (i=0; i<N; i++) {
for (j=0; j<N; j++)
printf("%d \t", matrix[i][j]);
printf ("\n");
}
printf ("\n");
}

void printMatrix2 (int matrix[N+2][N+2])
{
int i,j;
for (i=1; i<N+1; i++) {
for (j=1; j<N+1; j++)
printf("%d \t", matrix[i][j]);
printf ("\n");
}
printf ("\n");
}
