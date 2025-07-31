/*
Author: Polizois Siois 8535
*/
/*
Faculty of Electrical and Computer Engineering AUTH
4th assignment on Parallel and Distributed Systems (7th semester)
*/
/*
Parallel implementation of the PageRank algorithm.
Uses the Gauss-Seidel method to solve a linear system of equations.
The parallelism is implemented using OpenMP.

The algorithm accepts as input:
- The path that contains the data on which the algorithm will be applied.
The data needed are:
- A binary file containing an adjacency matrix of size (NxN) in sparse
representation.So the binary file actually contains an integer array
of 2 columns and rows as many as the non-zero elements of the full
adjacency matrix.The first column contains the row-number of the non-zero
element on the full adjacency matrix and the second column contains the
column-number of the element.
The matrix represents a directed graph with N nodes and edges defined by
the elements A[i][j] (directed edge from node i to node j).
- A text file containing a 1d array of size (N) and type (string).The
array contains the label of each node.
- A binary file containing a 1d array of size (N) and type (double).The
array contains the PageRanks that have been calculated on matlab for
the same dataset.It is used to verify that the results from this
algorithm are correct.
- The number of the nodes (N).
- The number of threads to be used for the paralellism.

The algorithm calculates the PageRank vector of the nodes and prints:
- The time taken for the algorithm to converge.
- The convergence error.
- The iterations taken for the algorithm to converge.
- The divergence of the calculated Pagerank vector from the matlab one.

References:
https://en.wikipedia.org/wiki/PageRank
https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method
https://www.ece.ucsb.edu/~hespanha/published/2018ACC_0753_MS.pdf
https://www.mathworks.com/help/matlab/examples/use-page-rank-algorithm-to-rank-websites.html
*/

// NOTE COMPILE
// gcc pr_omp_gs.c -o pr_omp_gs -lm -O3 -fopenmp

// NOTE
// Accessing matrixes line-by-line is WAY FASTER than column-by-column.
// In the data I use, an (i,j) cell of the adjacency indicates an outbound
// link from node-i to node-j.
// So in order to implement the algorithm, i would normally have to
// access the matrix column-by-column in each iteration.
// In order to make the implementation faster i have to take the transpose
// of the matrix and access it line-by-line.


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>


char *dataPath;
int nodeNum;
int NUM_THREADS;

#define labelSize 200
#define d 0.85         // dumping factor
#define err 0.0001     // maximum convergence error
#define maxIters 200   // maximum number of iterations
#define matrixFileName "sparseMatrix.bin"
#define labelsFileName "labels.txt"
#define resultsFileName "matResults.bin"

// Time variables
struct timeval startwtime, endwtime;

// Fuction declarations
double timer(char *command);
double** alloc_matrix_double(int rows, int cols);
char** alloc_matrix_char(int rows, int cols);
int fileLoad_bin(char *filePath, int rows, int cols, double **dest);
int sparseFileLoad_bin(char *filePath, double **dest);
int fileLoad_txt(char *filePath, int rows, size_t cols, char **dest);
void fillArray(double *arr, int size, double value);
void countOutLinks(double *arr, double **matrix, int size);
void transpose(double **matrix, int size);
void partSum(double *save, double *a, double *b, int start, int end);

double quad_err(double *a, double *b, int size);
double error(double **A, double *x, double b, int size);

double checkResults(char *matResults, double *pr);

// argv[1] = dataPath
// argv[2] = nodeNum
// argv[3] = NUM_THREADS
int main(int argc, char *argv[])
{
dataPath = argv[1]; nodeNum = atoi(argv[2]); NUM_THREADS = atoi(argv[3]);

// Variable Declaration
FILE *matrixFile;
FILE *labelsFile;
double **matrix; // (nodeNum)x(nodeNum)
char **labels;   // (nodeNum)x(200)
int i, j;
char matrixFilePath[100]; strcpy(matrixFilePath, dataPath); strcat(matrixFilePath, matrixFileName);
char labelsFilePath[100]; strcpy(labelsFilePath, dataPath); strcat(labelsFilePath, labelsFileName);
char resultsFilePath[100]; strcpy(resultsFilePath, dataPath); strcat(resultsFilePath, resultsFileName);


// Memory Allocation
matrix = alloc_matrix_double(nodeNum, nodeNum);     // Matrix
labels = alloc_matrix_char(nodeNum, labelSize);     // Labels
double *pr = malloc(nodeNum * sizeof(*pr));         // Latest PageRank Array
double *pr_old = malloc(nodeNum * sizeof(*pr_old)); // Previous iteration PageRank Array
double *outL = malloc(nodeNum * sizeof(*outL));     // Outbound Links Array
int *indexes = malloc(nodeNum * sizeof(*indexes));  // Outbound Links Array

// File Reading
sparseFileLoad_bin(matrixFilePath, matrix);               // Matrix
fileLoad_txt(labelsFilePath, nodeNum, labelSize, labels); // Labels

// Array Initialization
fillArray(pr, nodeNum, 1/(double)nodeNum); // initialize with 1/N
fillArray(outL, nodeNum, 0);               // initialize with 0
countOutLinks(outL, matrix, nodeNum);      // initialize with the outbound links of each node


// Normalize Matrix and multiply with the Dumping Factor
for(i=0;i<nodeNum;i++) // rows
for(j=0;j<nodeNum;j++) // cols
matrix[i][j] = matrix[i][j] * d / outL[i];

// Get the Transpose of the Matrix
transpose(matrix, nodeNum);

// Calculate (I - matrix)
for(i=1;i<nodeNum;i++)
for(j=0;j<i;j++) { matrix[i][j] = - matrix[i][j]; matrix[j][i] = - matrix[j][i]; }
for(i=0;i<nodeNum;i++) matrix[i][i] = 1 - matrix[i][i];

// So now we are at the point where we want to solve this system of linear equations:
// matrix'[(nodeNum)x(nodeNum)] * pr[(nodeNumx)(1)] = (1-b)/nodeNum[(nodeNumx)(1)]
// where matrix' is the original matrix after all the processing it's been through

// Notice that the system is in the A*x = b form which means that it can be solved
// using the Gauss-Seidel Gauss-Seidel method

// Gauss-Seidel Implementation of PageRank
// based on this paper: https://www.ece.ucsb.edu/~hespanha/published/2018ACC_0753_MS.pdf

double sum, conv, sum1, sum2;
int iters = 0;

omp_set_num_threads(NUM_THREADS); // Creating the user-demanded threads
omp_set_nested(0);                // Disabling nested parallelism because i noticed it makes the program slower

timer("tic"); // Starting timer
do
{
iters++;

// PAR
#pragma omp parallel for default(shared) private(i) schedule(auto)
for(i=0;i<nodeNum;i++) pr_old[i] = pr[i]; // Update the pr_old array with the new pageranks (needed for quad_err)

// PAR
#pragma omp parallel num_threads(NUM_THREADS) private(i, j, sum, sum1, sum2)
{
#pragma omp for schedule(auto) nowait
for(i=0;i<nodeNum;i++) // rows
{
sum=0; sum1 = 0; sum2 = 0;

// sum1+sum2
#pragma omp parallel num_threads(2) // If i enable nested parallelism it becomes slower
{
#pragma omp sections
{
#pragma omp section
partSum(&sum1, pr, matrix[i], 0, i);

#pragma omp section
partSum(&sum2, pr, matrix[i], i+1, nodeNum);
}
}

sum = sum1+sum2;

// Update the PageRank of node 'i'
pr[i] = 1/matrix[i][i] * ( (1-d)/nodeNum - sum);
}
}

// Check for convergence
// conv = error(matrix, pr, (1-d)/nodeNum, nodeNum); // 1st way to check convergence
conv = quad_err(pr, pr_old, nodeNum); // 2nd way to check convergence

}while( conv > err && iters < maxIters);

// Print statistics
printf("Convergence time: %f sec\n", timer("toc")); // Stop timer and print the convergence time
printf("Error: %f\n", conv);                        // Print the error of the convergence
printf("Number of iterations: %d\n", iters);        // Print the number of iterations until convergence
printf("Divergence from matlab results: %f\n", checkResults(resultsFilePath, pr)); // Check Results
printf("\n");

// Print Results
// for(i=0;i<25;i++) printf("%d: pr = %f :: outLinks = %d :: %s", i+1, pr[i], (int)outL[i], labels[i]);

// Free Allocated space
for(i=0;i<nodeNum;i++) free(matrix[i]); free(matrix);
free(pr);
free(pr_old);
free(outL);
free(indexes);

return 0;
}

// Loads data from bin file
int fileLoad_bin(char *filePath, int rows, int cols, double **dest)
{
FILE *file;
file=fopen(filePath,"rb");

int i;

for (i=0; i < rows; i++)
{
// Loading a row of the matrix
if (!fread(&dest[i][0], sizeof(double), cols, file))
{
printf("Unable to read Matrix from file!");
return 1;
}
}

fclose(file);
return 0;
}

// Loads data from bin file(sparse representation) to a full matrix
int sparseFileLoad_bin(char *filePath, double **dest)
{
FILE *file;
file=fopen(filePath,"rb");
int temp[2];

while(fread(&temp[0], sizeof(int), 2, file))
{
dest[temp[0]-1][temp[1]-1] = 1;
}

fclose(file);
return 0;
}

// Loads data from txt file
int fileLoad_txt(char *filePath, int rows, size_t cols, char **dest)
{
FILE *file;
file=fopen(filePath,"r");

int i;

for (i=0; i < rows; i++)
{
// Loading a line of text (label)
if(getline(&dest[i], &cols, file) == -1)
{
printf("Unable to read Labels from file!");
return 1;
}
}

fclose(file);
return 0;
}

// Allocates memory for a 2d array of doubles
double** alloc_matrix_double(int rows, int cols)
{
int i;

double **matrix= malloc(rows * sizeof(*matrix));
if(!matrix)
{
printf("Out of memory\n");
exit(-1);
}

for(i=0;i<rows;i++)
{
matrix[i] = malloc(cols * sizeof(**matrix));
if(!matrix[i])
{
printf("Out of memory\n");
exit(-1);
}
}

return matrix;
}

// Allocates memory for a 2d array of chars
char** alloc_matrix_char(int rows, int cols)
{
int i;

char **matrix= malloc(rows * sizeof(*matrix));
if(!matrix)
{
printf("Out of memory\n");
exit(-1);
}

for(i=0;i<rows;i++)
{
matrix[i] = malloc(cols * sizeof(**matrix));
if(!matrix[i])
{
printf("Out of memory\n");
exit(-1);
}
}

return matrix;
}

// Fills a given 1d array (arr) with a given value
void fillArray(double *arr, int size, double value)
{
for(int i=0;i<size;i++) arr[i] = value;
}

// Counts the outbound links of each node of a given adjacency matrix(2d)
// and stores them in arr(1d)
void countOutLinks(double *arr, double **matrix, int size)
{
int i, j;
for(i=0;i<nodeNum;i++)
{
for(j=0;j<nodeNum;j++)
if(matrix[i][j]) arr[i]++;

if(!arr[i])
{
for(j=0;j<nodeNum;j++) matrix[i][j] = 1;
arr[i] = nodeNum;
}
}
}

// Calculates  the norm of a 1d array of doubles
double quad_err(double *a, double *b, int size)
{
double sum = 0;
int i;

#pragma omp parallel for default(shared) private(i) schedule(auto) reduction(+:sum)
for(i=0;i<size;i++) sum = sum + (a[i]-b[i])*(a[i]-b[i]);

return sqrt(sum);
}

// Calculates the norm |Ax-b|
double error(double **A, double *x, double b, int size)
{
double sum = 0, er = 0;
for(int i=0; i<size; i++)
{
sum = 0;
for(int j=0;j<size;j++)
sum += A[i][j]*x[j];
sum -= b;

er += sum*sum;
}

return sqrt(er);
}

// use timer("tic") to start the timer and get the curent time
// use timer("toc") to get the time between this toc and the last tic
double timer(char *command)
{
double gap = -1;

if(command == "tic")
{
gettimeofday( &startwtime, NULL ); // Start timer
endwtime.tv_sec = 0;
endwtime.tv_usec = 0;
gap = (double)( startwtime.tv_usec / 1.0e6 + startwtime.tv_sec );
}
else if(command == "toc")
{
gettimeofday( &endwtime, NULL );   // Stop timer
gap = (double)( ( endwtime.tv_usec - startwtime.tv_usec ) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec );
}

return gap;
}

// Turns a 2d (double) matrix of (size)x(size) size into its Transpose
void transpose(double **matrix, int size)
{
double temp;

for(int i=1;i<size;i++)
{
for(int j=0;j<i;j++)
{
temp = matrix[i][j];
matrix[i][j] = matrix[j][i];
matrix[j][i] = temp;
}
}
}

void partSum(double *save, double *a, double *b, int start, int end)
{
double su = 0;
// #pragma omp parallel for default(shared) private(j) schedule(auto) reduction(+:su)
for(int j=start;j<end;j++) su += a[j] * b[j];

*save = su;
}

// Calculates and returns the distance between the calculated pagerank vector and
// the one calculated by matlab
double checkResults(char *matResults, double *pr)
{
double *prMat = malloc(nodeNum * sizeof(*prMat));
fileLoad_bin(matResults, 1, nodeNum, &prMat);

return quad_err(prMat, pr, nodeNum);
}
