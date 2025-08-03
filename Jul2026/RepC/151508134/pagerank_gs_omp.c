#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#define alpha 0.85
#define ERR 0.0001
#define MAXITERS 200


double** Mcalloc(int rows, int cols);
void Mfree(double** matrix, int rows);
int getNumberOfnodes(char *path);
int* loadAdjMat(char *path, double **matrix, int rows);
void transpose(double **matrix, int rows);   // assuming sqare matrix
void getPagerankMatrix(double **matrix, int *outdeg, int rows);
void savePageranks(char *path, double *prank, int size);


int main(int argc, char const *argv[]) {
int Nnodes;
int *outdeg;
double **A;   //  adjacency matrix
double *x;    //  pagerank matrix
char *path;   //  path to directory of files
int num_threads = 2;

/*
* The argument for the program is the directory name
* of the query for which we want to create the adjacency matrix
*/
if (argc < 2){
printf("ERROR: Enter at least only ONE argument\n");
printf("first argument: directory of file nodes\n");
printf("second argument: number of threads\n");
exit(1);
} else if (argc == 3) {
num_threads = atoi(argv[2]);
} else if (argc > 3) {
printf("ERROR: Ented TOO many arguments!\n");
printf("first argument: directory of file nodes\n");
printf("second argument: number of threads\n");
exit(1);
}
path = strdup(argv[1]);

Nnodes = getNumberOfnodes(path);
A = Mcalloc(Nnodes, Nnodes);
x = (double*)malloc(Nnodes * sizeof(double));
if (x == NULL) { printf("ERROR: Cant allocate memory for x\n"); exit(1);}


outdeg = loadAdjMat(path, A, Nnodes);
for (size_t i = 0; i < Nnodes; i++) x[i] = 1.0 / Nnodes;
getPagerankMatrix(A, outdeg, Nnodes);

int iters = 0;
int i, j;
double dsum = 0;
double dot, x_new;

struct timeval startwtime, endwtime;
double seq_time;

omp_set_num_threads(num_threads);
omp_set_nested(0);

gettimeofday (&startwtime, NULL);
do {
iters++;
dsum = 0;

#pragma omp parallel for schedule(auto) private(j,dot,x_new) reduction(+:dsum)
for ( i = 0; i < Nnodes; i++) {
dot = 0;

for ( j = 0; j < Nnodes; j++) {
if (i != j) dot += A[i][j] * x[j];
}

x_new = ((1-alpha)/Nnodes - dot) / A[i][i];
dsum += (x[i] - x_new) * (x[i] - x_new);
x[i] = x_new;
}
} while(sqrt(dsum) > ERR && iters < MAXITERS);

gettimeofday (&endwtime, NULL);
seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);

//for (size_t i = 0; i < 25; i++) printf("x =\t%f\n", x[i]);
printf("time:\t%f epsilon:\t%f it:\t%d\n", seq_time, sqrt(dsum), iters);
//printf("epsilon :\t%f\n", sqrt(dsum));
//printf("iterations :\t%d\n", iters);

savePageranks(path, x, Nnodes);

Mfree(A, Nnodes);
free(outdeg);
free(x);
free(path);

return 0;
}


/**
* allocates a zero initialized double matrix
* @method Mcalloc
* @param  rows    number of rows
* @param  cols    number of columns
* @return         pointer to the first address of matrix
*/
double** Mcalloc(int rows, int cols) {
double** matrix = (double**)malloc(rows * sizeof(double*));
if (matrix == NULL) {
printf("ERROR: Can't allocate memory for matrix!\n");
exit(1);
}
for (size_t i = 0; i < rows; i++) {
matrix[i] = (double*)calloc(cols, sizeof(double*));
if (matrix[i] == NULL) {
printf("ERROR: Can't allocate memory for matrix!\n");
exit(1);
}
}
return matrix;
}


/**
* frees from memory a row major matrix
* @method Mfree
* @param  pointer to the first address of matrix
* @param  rows   number of rows of matrix
*/
void Mfree(double **matrix, int rows) {
for (size_t i = 0; i < rows; i++) {
free(matrix[i]);
}
free(matrix);
}


/**
* Opens file "nodes" and reads the number of nodes in the graph of pages.
* This function should be used in conjuction with funcion loadAdjMat().
* @method getNumberOfnodes
* @param  path             the directory whre the file is located
* @return                  number of nodes(pages) in file "adj_list"
*/
int getNumberOfnodes(char *path) {
FILE *fnodes;
char nodes_file[1000];
int Nnodes;

sprintf(nodes_file, "%s/nodes", path);
fnodes = fopen(nodes_file, "r");
if (fnodes == NULL) {
printf("ERROR: Can't open file %s\n", nodes_file);
exit(1);
}
fscanf(fnodes, "%d", &Nnodes);
fclose(fnodes);

return Nnodes;
}


/**
* loads from a file the adjacency matrix of a graph of pages, and returns the
* outdegree of the pages.
* The file MUST be named "adj_list" and each entry of the list MUST be in the
* form  pid: pid1,pid2,.....,pidN,-1 , which means that the page with id pid,
* points to the pages with ids pid1,pid2,.....,pidN.
* if there are pages with no outbound links, then links are added to theese
* pages pointing to ALL the pages, includint itself.
* @method loadAdjMat
* @param  path       the directory whre the file is located
* @param  matrix     the pointer to adjacency matrix
* @param  rows       rows of the matrix
* @return            pointer to an array of outdegrees of the adjacency matrix
*/
int* loadAdjMat(char *path, double **matrix, int rows) {
FILE *flist;
char list_file[1000];
int j;
int *outdeg;

sprintf(list_file,"%s/adj_list",path);
flist = fopen(list_file, "r");
if (flist == NULL) {
printf("ERROR: Can't open file %s\n", list_file);
exit(1);
}

outdeg = (int*)calloc(rows, sizeof(int));
if (outdeg == NULL) {
printf("Error: Cannot allocate memory for outdegree array!\n");
exit(1);
}
for (size_t i = 0; i < rows; i++) {
fscanf(flist, "%*d: %d", &j);
while (j != -1) {
matrix[i][j] = 1.0;     // if there is a link from i to j set value to 1.0
outdeg[i]++;            // if there is a link from i to j increase outdegree of i by one
fscanf(flist, "%d", &j);
}
}
fclose(flist);

//  check for dangling nodes
for (size_t i = 0; i < rows; i++) {
if (outdeg[i] == 0) {
for (size_t j = 0; j < rows; j++) matrix[i][j] = 1;
outdeg[i] = rows;
}
}

return outdeg;
}


/**
* Transposes a given square matrix.
* @method transpose
* @param  matrix    square matrix to be transposed
* @param  rows      number of rows of the square matrix
*/
void transpose(double **matrix, int rows) {
double temp;

for (size_t i = 0; i < rows; i++) {
for (size_t j = 0; j < i; j++) {
temp = matrix[i][j];
matrix[i][j] = matrix[j][i];
matrix[j][i] = temp;
}
}
}


/**
* Transforms a square row stochastic matrix in the nedded format for the
* Gauss-Seidel pagerank algorithm.
* The algorithm and the matrix format are described in this paper:
* https://www.ece.ucsb.edu/~hespanha/published/2018ACC_0753_MS.pdf
* Specifically the needed matrix is (I-(1-m)A) in p.3 eq.3 of the same paper.
*
* @method getPagerankMatrix
* @param  matrix            pointer to square row stochastic matrix
* @param  outdeg            pointer to outdegree array of the matrix
* @param  rows              rows of the matrix and lenth of outdegree array
*/
void getPagerankMatrix(double **matrix, int *outdeg, int rows) {

for (size_t i = 0; i < rows; i++) {
for (size_t j = 0; j < rows; j++) {
matrix[i][j] = alpha * matrix[i][j] / outdeg[i];
}
}

transpose(matrix, rows);

for (size_t i = 0; i < rows; i++) {
for (size_t j = 0; j < i; j++) {
matrix[i][j] = -matrix[i][j];
matrix[j][i] = -matrix[j][i];
}
}

for (size_t i = 0; i < rows; i++) {
matrix[i][i] = 1-matrix[i][i];
}
}


void savePageranks(char *path, double *prank, int size) {
FILE *fpranks;
char pranks_file[1000];

sprintf(pranks_file,"%s/pageranks_omp",path);
fpranks = fopen(pranks_file, "w");
if (fpranks == NULL){
printf("ERROR: Cant open file %s\n didt save resalts\n",pranks_file);
exit(1);
}

for (size_t i = 0; i < size; i++) {
fprintf(fpranks, "%f\n", prank[i]);
}
}
