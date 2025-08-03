#include <stdio.h>
#include<stdlib.h>
#include <omp.h>

void initialize(int nra,int *a)
{
int i,j,k;
for (i=0; i<nra; i++)
for (j=0; j<nra; j++)
a[(0*nra*nra)+(i*nra+j)]= i+j; //matrix A and B are filled with random order number
for (i=0; i<nra; i++)
for (j=0; j<nra; j++)
a[(1*nra*nra)+(i*nra+j)]= i*j;
for (i=0; i<nra; i++)
for (j=0; j<nra; j++)
a[(2*nra*nra)+(i*nra+j)]= 0; //matrix C is initializing with zeros
}
void row_column(int nra,int *a,int chunk,int nthreads) //Function to implement the row-column product
{

int i,j,k;
#pragma omp parallel shared(a,chunk,nthreads) private(i,j,k)
{
#pragma omp for schedule (dynamic, chunk)
for (i = 0; i < nra; i++)
{
for (j = 0; j < nra; j++)
for (k = 0; k < nra; k++)
a[(2 * nra * nra) + (i * nra + j)] += a[(0 * nra * nra) + (i * nra + k)] *
a[(1 * nra * nra) + (k * nra + j)]; //product row-column
}
}

}
void transpose(int nra,int *a,int chunk,int nthreads) //this function compute the transpose of the matrix in a way to allow a multiplication between two vectors.
{

int i, j;
int *bT= malloc(nra*nra*sizeof(int)); // Matrix used like a Temp matrix: to not lose the elements doing the transposition
#pragma omp parallel shared(a,bT,chunk,nthreads) private(i,j)
{
#pragma omp for schedule (dynamic, chunk)
for (i = 0; i < nra; ++i) {
for (j = 0; j < nra; ++j) {
bT[j * nra + i] = a[1 * nra * nra + (i * nra + j)];
}
}
#pragma omp for schedule (dynamic, chunk)
for (i = 0; i < nra; ++i) {
for (j = 0; j < nra; ++j) {
a[1 * nra * nra + (i * nra + j)]=bT[i * nra + j];
}
}

}
row_column(nra, a, chunk,nthreads);
free(bT);
}



//function to subdivide matrix
void divide(int nca,int *a,int *ABC,int num,int nthreads,int chunk)
{
int i,j,k;
#pragma omp parallel shared(a,ABC,nthreads,chunk) private(i,j)

#pragma omp for schedule (dynamic, chunk)
for (i = 0; i < 2; i++)
{
for (j = 0; j < 2; j++)
{
for(k=0;k<nca/2;k++)
{
for(int d=0;d<nca/2;d++)
{
ABC[num+(2*(nca/2)*(nca/2)*i+(nca/2)*(nca/2)*j+(nca/2)*k+d)]=a[num+((k+(nca/2)*i)*(nca)+(d+(nca/2)*j))];
}
}

}

}

}
//function to sum matrix
void sum(int nca,int in1, int in2,int *ABC,int in3,int *res,int nthreads,int chunk)
{
int i,j;
#pragma omp parallel shared(ABC,res,in1,in2,nthreads,chunk) private(i,j)
{
#pragma omp for schedule (dynamic, chunk)
for (i = 0; i < nca/2; i++) {
for (j = 0; j < nca / 2; j++) {
res[in3+(i * (nca / 2) + j)] = ABC[in1+(i * (nca / 2) + j)] + ABC[in2+(i * (nca / 2) + j)];
}
}
}
}
//function to subtract matrix
void sub(int nca,int in1, int in2,int *ABC,int in3,int *res,int nthreads,int chunk)
{
int i,j;
#pragma omp parallel shared(ABC,res,in1,in2,nthreads,chunk) private(i,j)
{
#pragma omp for schedule (dynamic, chunk)
for (i = 0; i < nca/2; i++) {
for (j = 0; j < nca / 2; j++) {
res[in3+(i * (nca / 2) + j)] = ABC[in1+(i * (nca / 2) + j)] - ABC[in2+(i * (nca / 2) + j)];
}
}
}
}
//function to assign to the final matrix the 4 sub matrix
void assigntoA(int nca,int *a,int * ABC,int in1,int in2,int in3,int in4,int nthreads,int chunk)
{
int i,j;
#pragma omp parallel shared(a,ABC,nthreads,chunk) private(i,j)
{
#pragma omp for schedule (dynamic, chunk)
for (i = 0; i < nca / 2; i++)
{
for (j = 0; j < nca / 2; j++)
{
a[(2*nca*nca)+(i * (nca) + j)] = ABC[in1+(i * (nca / 2) + j)];
a[(2*nca*nca)+(i * (nca) + j + nca / 2)] = ABC[in2+(i * (nca / 2) + j)];
a[(2*nca*nca)+((i + nca / 2) * (nca) + j)] = ABC[in3+(i * (nca / 2) + j)];
a[2*nca*nca+((i + nca / 2) * (nca) + j + nca / 2)] = ABC[in4+(i * (nca / 2) + j)];
}
}
}
}
//Function to compute the product between intermediate matrix; this is a redifinition of the row column function because the index are different
void productBetweenRes(int nca,int in1,int *resA,int in2, int *resB,int *m,int inm,int nthreads,int chunk)
{
int i,j,k;
#pragma omp parallel shared(resA,resB,m,inm,nca,nthreads,chunk) private(i,j,k)
{
#pragma omp for schedule (dynamic, chunk)
for (i=0; i<nca/2; i++)
{
for(j=0; j<nca/2; j++)
for (k=0; k<nca/2; k++)
m[inm+(i*nca/2+j)] += resA[in1+(i*(nca/2)+k)] * resB[in2+(k*(nca/2)+j)];
}

}
}
//Used to make sum between intermediate matrix
void sumBetweenSub(int nca,int in1, int*A,int in2,int *B,int in3,int *C,int nthreads,int chunk)
{
int i,j;
#pragma omp parallel shared(A,B,C,in1,in2,nthreads,chunk) private(i,j)
{
#pragma omp for schedule (dynamic, chunk)
for (i = 0; i < nca/2; i++) {
for (j = 0; j < nca / 2; j++) {
C[in3+(i * (nca / 2) + j)] = A[(in1+(i * (nca / 2) + j))] + B[in2+(i * (nca / 2) + j)]; 
}
}
}
}

//Used to make difference between intermediate matrix
void subBetweenSub(int nca,int in1, int*A,int in2,int *B,int in3,int *C,int nthreads,int chunk)
{
int i,j;
#pragma omp parallel shared(A,B,C,in1,in2,nthreads,chunk) private(i,j)
{
#pragma omp for schedule (dynamic, chunk)
for (i = 0; i < nca/2; i++) {
for (j = 0; j < nca / 2; j++) {
C[in3+(i * (nca / 2) + j)] = A[(in1+(i * (nca / 2) + j))] - B[in2+(i * (nca / 2) + j)];
}
}
}
}

void strassen(int nca,int *a,int chunk,int nthreads) // strassen Method
{
int (*ABC)= calloc(12*(nca/2)*(nca/2),sizeof(int)); //define the 12 submatrix
int (*M)= calloc(9*(nca/2)*(nca/2),sizeof(int)); //define the 9 matrix to store the intermediate results
//these 3 call to divide function divide the matrix a,b and c in 4 submatrix
divide(nca,a,ABC,0,nthreads,chunk);
divide(nca,a,ABC,(nca*nca),nthreads,chunk);
divide(nca,a,ABC,2*(nca*nca),nthreads,chunk);

//from now intermediate operations
sum(nca,0,3*(nca/2)*(nca/2),ABC,7*(nca/2)*(nca/2),M,nthreads,chunk);
sum(nca,4*(nca/2)*(nca/2),7*(nca/2)*(nca/2),ABC,8*(nca/2)*(nca/2),M,nthreads,chunk);
productBetweenRes(nca,7*(nca/2)*(nca/2),M,8*(nca/2)*(nca/2),M,M,0*(nca/2)*(nca/2),nthreads,chunk);

sum(nca,2*(nca/2)*(nca/2),3*(nca/2)*(nca/2),ABC,7*(nca/2)*(nca/2),M,nthreads,chunk);
productBetweenRes(nca,7*(nca/2)*(nca/2),M,4*(nca/2)*(nca/2),ABC,M,(nca/2)*(nca/2),nthreads,chunk);

sub(nca,5*(nca/2)*(nca/2),7*(nca/2)*(nca/2),ABC,8*(nca/2)*(nca/2),M,nthreads,chunk);
productBetweenRes(nca,0,ABC,8*(nca/2)*(nca/2),M,M,2*(nca/2)*(nca/2),nthreads,chunk);

sub(nca,6*(nca/2)*(nca/2),4*(nca/2)*(nca/2),ABC,8*(nca/2)*(nca/2),M,nthreads,chunk);
productBetweenRes(nca,3*(nca/2)*(nca/2),ABC,8*(nca/2)*(nca/2),M,M,3*(nca/2)*(nca/2),nthreads,chunk);

sum(nca,0*(nca/2)*(nca/2),1*(nca/2)*(nca/2),ABC,7*(nca/2)*(nca/2),M,nthreads,chunk);
productBetweenRes(nca,7*(nca/2)*(nca/2),M,7*(nca/2)*(nca/2),ABC,M,4*(nca/2)*(nca/2),nthreads,chunk);

sub(nca,2*(nca/2)*(nca/2),0*(nca/2)*(nca/2),ABC,7*(nca/2)*(nca/2),M,nthreads,chunk);
sum(nca,4*(nca/2)*(nca/2),5*(nca/2)*(nca/2),ABC,8*(nca/2)*(nca/2),M,nthreads,chunk);
productBetweenRes(nca,7*(nca/2)*(nca/2),M,8*(nca/2)*(nca/2),M,M,5*(nca/2)*(nca/2),nthreads,chunk);

sub(nca,1*(nca/2)*(nca/2),3*(nca/2)*(nca/2),ABC,7*(nca/2)*(nca/2),M,nthreads,chunk);
sum(nca,6*(nca/2)*(nca/2),7*(nca/2)*(nca/2),ABC,8*(nca/2)*(nca/2),M,nthreads,chunk);
productBetweenRes(nca,7*(nca/2)*(nca/2),M,8*(nca/2)*(nca/2),M,M,6*(nca/2)*(nca/2),nthreads,chunk);


sumBetweenSub(nca,0*(nca/2)*(nca/2),M,3*(nca/2)*(nca/2),M,8*(nca/2)*(nca/2),ABC,nthreads,chunk);
subBetweenSub(nca,8*(nca/2)*(nca/2),ABC,4*(nca/2)*(nca/2),M,8*(nca/2)*(nca/2),ABC,nthreads,chunk);
sumBetweenSub(nca,8*(nca/2)*(nca/2),ABC,6*(nca/2)*(nca/2),M,8*(nca/2)*(nca/2),ABC,nthreads,chunk);

sumBetweenSub(nca,2*(nca/2)*(nca/2),M,4*(nca/2)*(nca/2),M,9*(nca/2)*(nca/2),ABC,nthreads,chunk);

sumBetweenSub(nca,1*(nca/2)*(nca/2),M,3*(nca/2)*(nca/2),M,10*(nca/2)*(nca/2),ABC,nthreads,chunk);

subBetweenSub(nca,0*(nca/2)*(nca/2),M,1*(nca/2)*(nca/2),M,11*(nca/2)*(nca/2),ABC,nthreads,chunk);
sumBetweenSub(nca,11*(nca/2)*(nca/2),ABC,2*(nca/2)*(nca/2),M,11*(nca/2)*(nca/2),ABC,nthreads,chunk);
sumBetweenSub(nca,11*(nca/2)*(nca/2),ABC,5*(nca/2)*(nca/2),M,11*(nca/2)*(nca/2),ABC,nthreads,chunk);
// this function assign to the final matrix the submatrix
assigntoA(nca,a,ABC,8*(nca/2)*(nca/2),9*(nca/2)*(nca/2),10*(nca/2)*(nca/2),11*(nca/2)*(nca/2),nthreads,chunk);


}


int main()
{
int nra,i,j;
double begin,end,time_spent;
printf("Insert ROW of A: "); //I/O requests for the matrix dimension
scanf("%d",&nra);
int chunk=1;
int var;
printf("Inserire numero Threads: ");
scanf("%d",&var);
omp_set_num_threads(var);
int *a= malloc(3*nra*nra*sizeof(int)); // only one big matrix contains all the 3 matrices
int nthreads=omp_get_num_threads();

initialize(nra,a);// initialize the matrices

/*NAIVE METHOD*/
begin=omp_get_wtime(); //timer starts
row_column(nra,a,chunk,nthreads); // Naive product (row-column)
end= omp_get_wtime(); //timer ends
time_spent = (end - begin);
printf("\nNaive Time Spent: %lf\n",time_spent);//Time Computing

/*TRANSPOSE METHOD*/
initialize(nra,a);
begin=omp_get_wtime(); //timer starts
transpose(nra,a,chunk,nthreads); // Naive product (row-column)
end= omp_get_wtime(); //timer ends
time_spent = (end - begin); //Time Computing
printf("\nTranspose Time Spent: %lf\n",time_spent);

//Strassen Method
initialize(nra,a);
begin=omp_get_wtime();
strassen(nra,a,chunk,nthreads); //Strassen Method, constrain: square matrix 2^n x 2^n
end = omp_get_wtime();
time_spent =(end - begin);
printf("\nStrassen Time Spent: %lf\n",time_spent);

}
