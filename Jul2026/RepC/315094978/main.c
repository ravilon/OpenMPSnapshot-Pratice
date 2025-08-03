#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>


int main(int argc, char **argv) {

int i, j;

double start, end; //check time
double globalStart, globalEnd;


int **A, **B;

int N; //size for matrix A[N][N], B[N][N]

int t; //the number of threads

int chunk; //the size of lines each thread has to work with

int sum; //a temp sum that the threads use to store the row's sum values except the diagonal one

int max; //max abs diagonal value of A matrix 

int localMin; //localMin is used from each thread to store the min value of each chunk

int minLocalIndexI, minLocalIndexJ; //local indexes for min value of B matrix

int min; //min value of B matrix

int  minIndexI, minIndexJ; //indexes for min value of B matrix

bool isSDD = true; //a bool value used to determine if the A matrix is a Strictly diagonally dominant


printf("Give size of matrix: ");
scanf("%d", &N);

printf("Give number of Thread: ");
scanf("%d", &t);

//set number of thread
omp_set_num_threads(t);

//malloc array
A = malloc(sizeof(int*)*N);
B = malloc(sizeof(int*)*N);
for(i = 0 ; i < N; i++){
A[i] = malloc(sizeof(int)*N);
B[i] = malloc(sizeof(int)*N);
}

//read array
for (i = 0; i < N; i++) {
for (j = 0; j < N; j++) {
printf("Give the A[%d][%d]: ", i+1, j+1);
scanf("%d", &A[i][j]);
}
}
globalStart = omp_get_wtime();
start = omp_get_wtime();
#pragma omp parallel shared(A, N, chunk, isSDD) private(i, j, sum) 
{

chunk = N / omp_get_num_threads();

#pragma omp for schedule(dynamic, chunk)
for(i = 0; i < N; i++) {

//if a thread has found that the condition is not meet
//skip the rest loops
if(!isSDD) {
continue;
}

//init sum for every line
sum = 0;

for(j = 0; j < N; j++) {
if(i != j) {
sum += abs(A[i][j]);
}
}

//check the condition for strictly diagonally approaching
if(abs(A[i][i]) <= sum) {
//if 2 or more threads try to change the value of isSCD its going to be the value "false"
//no critical condition
isSDD = false;
}
}
}
end = omp_get_wtime();
printf("\nA.a Threads %d, Time: %.4f\n", t, end - start);
if(isSDD) {
printf("\nThe A matrix is a strictly diagonally dominant\n");
start = omp_get_wtime();
#pragma omp parallel for private(i) reduction(max: max)
for (i = 0; i < N; i++) {
if (max < abs(A[i][i]))
max = abs(A[i][i]);
}
end = omp_get_wtime();
printf("\nA.b Threads %d, Time: %.4f\n", t, end - start);
printf("\nThe max value is: %d\n", max);

//create B array based on max value
start = omp_get_wtime();
#pragma omp parallel shared(A, B, N, chunk, max) private(i, j)
{
chunk = N / omp_get_num_threads();
#pragma omp for schedule(static, chunk)
for(i = 0; i < N; i++){
for(j = 0; j < N; j++){
if(i != j){
B[i][j] = max - abs(A[i][j]);
}
else{
B[i][j] = max;
}
}
}
}
end = omp_get_wtime();
printf("\nA.c Threads %d, Time: %.4f\n", t, end - start);


printf("\nArray B: \n");
for(i = 0; i < 5; i++){
for(j = 0; j < 5; j++){
printf(" %d", B[i][j]);
}
printf("\n");
}


start = omp_get_wtime();
#pragma omp parallel shared(B, N, chunk, min, minIndexI, minIndexJ) private(localMin, minLocalIndexI, minLocalIndexJ, i, j)
{

chunk = N / omp_get_num_threads();
min = localMin = B[0][0];
minIndexI = minLocalIndexI = 0;
minIndexJ = minLocalIndexJ = 0;
#pragma omp for schedule(static, chunk)
//init min and localMin
for(i = 0; i < N; i++) {
for(j = 0; j < N; j++) {

if (B[i][j] < localMin){
localMin = B[i][j];
minLocalIndexI = i;
minLocalIndexJ = j;
}
}
}
#pragma omp critical (find_min)
{
if (localMin < min){
min = localMin;
minIndexI = minLocalIndexI;
minIndexJ = minLocalIndexJ;
}
}

}
end = omp_get_wtime();
printf("\nA.d Threads %d, Time: %.4f\n", t, end - start);

printf("\nThe min is B[%d][%d] = %d\n", minIndexI, minIndexJ, min);
}
else {
//if the matrix is not a SDD then, the below message is printed and the programm ends
printf("\nThe A matrix is not a strictly diagonally dominant\n");
}
globalEnd = omp_get_wtime();
printf("\nAll program Threads %d, Time: %.4f\n", t, globalEnd - globalStart);
//free memory
for(i=0;i<N;i++) {
free(A[i]);
free(B[i]);
}
free(A);
free(B);

return 0;
}