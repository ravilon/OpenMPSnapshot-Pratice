// Author:        Rachna Sidana
// Description:   This program applies Gaussian Elimination to find the solution of Linear Systems of Equations. It also
//                calculates the execution time of code that has been parallelized.
//                MPI + OpenMp


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>
#define SIZE 100000000
#define PROCSIZE 50000000
#define N 1000
 
int p;              // no of processors 
int matrixsize=N;     // size of matrix 
int pid;            // processor id 
int row_nos[PROCSIZE]; 
int no_in_proc;     // number of rows in an individual processor
struct timeval start_time, stop_time, elapsed_time;  // timers
  
void initialise_the_matrices();
void enterint(int *var);
void entermatrixcyc(int n, int p, float var[]); 
void output(float values[]);
void gauss(float m[], float x[]);
void deal_with_jth_row(float matrix[], int j);
void put_0_in_jth_pos(float rowj[], int j, float row[], int rowno);
void solve_ith_eqn_init (float *sum, float row[]);
void solve_ith_eqn_g (int j, float *sum, float row[], float new);
void solve_ith_eqn_h (int rowno, float *sum, float row[]);
void pipeline (void (*init) (float *, float *),void (*g) (int, float *, float *, float),void (*h) (int, float *, float *),float local_vals[], float res[]);
void makecyclic(float inarray[], float outarray[]);
void makecyclicmatrix(float inarray[], float outarray[]);
int mod(int x, int y);
float A_Orig[N][N],c,partialSum,X_Original[N],x[N], B1_Orig[N], sum=0.0, mvSum=0.0;
int n = N;
int main(int argc, char *argv[])
{
  int errcode,row,i,j, maxblocksize;
  float m[PROCSIZE*(SIZE+1)]; 
  float result[PROCSIZE]; /* the solution */
  errcode = MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &p);
  MPI_Comm_rank (MPI_COMM_WORLD, &pid);
  enterint(&matrixsize);
  i=0;
  for(row=pid; row<=matrixsize; row=row+p)
    {
      row_nos[i] = row+1;
      i++;
    }

  if (matrixsize%p == 0)
    maxblocksize = matrixsize/p;
  else
    maxblocksize = matrixsize/p+1;
  if (pid<=(matrixsize-1)%p)
    no_in_proc = maxblocksize;
  else
    no_in_proc = maxblocksize-1;
  entermatrixcyc(matrixsize, p, m); 
  double  numFlops;
  float gflops;
  gettimeofday(&start_time,NULL);
  gauss(m, result); 
  if (0 == pid) 
  {
    gettimeofday(&stop_time,NULL);
    timersub(&stop_time, &start_time, &elapsed_time); // Unix time subtract routine
    printf("\n\nTotal time was %f seconds.\n", elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0);
    numFlops = ((2.0f*N*N*N/3.0f)+(3.0f*N*N/2.0f)-(13.0f*N/6.0f));
    float flops = numFlops/(elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0);
    gflops = flops/1000000000.0f;
    printf("GFlops :  %f .\n",gflops);
   } 
  errcode = MPI_Finalize ();
}

void enterint(int *var)
{
  if (pid == 0)
  {
    *var=N;
  }
  MPI_Bcast(var, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

/*********************************************************************************************************************
It creates an Augmented matrix ( matrix A and Vector B )and then
distributes the rows of the Augmented Matrix in a cyclic fashion to all the processes.
**********************************************************************************************************************/

void entermatrixcyc(int n, int p, float var[])
{
  float input[SIZE*(SIZE+1)], cyclic[SIZE*(SIZE+1)]; 
  int i,j,count,no_rows;
  int sizes[p], displs[p];  
  if (pid == 0)
  {
    initialise_the_matrices();
    count = 0;
    for (i=1; i<=n; i++)
    {
    for (j=1;j<=n; j++)
    {
      input[count]=A_Orig[i][j];  
      count++;
    }
    count++;
    }
    for (i=1; i<=n; i++)
    {
      input[i*n+i-1]=B1_Orig[i];    
    }
    makecyclicmatrix(input, cyclic);
  }
  for (i=0;i<n%p;i++)
    sizes[i] = (n/p + 1)*(n+1);
  for (i=n%p;i<p;i++)
    sizes[i] = (n/p)*(n+1);
  displs[0] = 0;
  for (i=1;i<p;i++)
    displs[i] = displs[i-1] + sizes[i-1];
  MPI_Scatterv(cyclic, sizes, displs, MPI_FLOAT, var, SIZE*SIZE, 
               MPI_FLOAT, 0, MPI_COMM_WORLD);
}

void output(float values[])
{
  int elt;
  if (pid == 0)
    printf("\n\nThe solution vector is x = \n [");
  for (elt = 0; elt < matrixsize-1; elt++)
  {
    if (pid == elt%p)
    {
      printf("%.2f, ", values[elt/p]);
      fflush(NULL); 
    }
    MPI_Barrier (MPI_COMM_WORLD); 
  }
  if (pid == (matrixsize-1)%p)
    printf("%.2f]\n", values[(matrixsize-1)/p]);
}

/*********************************************************************************************************************
Gaussian Elimination
**********************************************************************************************************************/

void gauss(float m[], float x[])
{
  int j;
  for (j=1; j<matrixsize; j++)
    deal_with_jth_row(m, j);
  pipeline(solve_ith_eqn_init, solve_ith_eqn_g, solve_ith_eqn_h, m, x);
}


/*********************************************************************************************************************
reduction to upper triangular form
**********************************************************************************************************************/

void deal_with_jth_row(float matrix[], int j)

{

  MPI_Comm_size (MPI_COMM_WORLD, &p);
  MPI_Comm_rank (MPI_COMM_WORLD, &pid);
  float rowj[SIZE+1];
  int i;
  if (pid == mod(j-1,p))
      for(i=0;i<matrixsize+1;i++)
  rowj[i] = matrix[((j-1)/p)*(matrixsize+1)+i];
  MPI_Bcast(rowj, matrixsize+1, MPI_FLOAT, mod(j-1,p), MPI_COMM_WORLD);
  for (i=0;i<no_in_proc;i++)
    put_0_in_jth_pos(rowj, j, &matrix[i*(matrixsize+1)], row_nos[i]);
}

void put_0_in_jth_pos(float rowj[], int j, float row[], int rowno)
{
  int i;
  float multiplier;
  multiplier = -(row[j-1]/rowj[j-1]);
  if (rowno > j)
    for (i=0; i<matrixsize+1; i++)
      row[i] = multiplier*rowj[i] + row[i];
}

/*********************************************************************************************************************
back substitution equation:
**********************************************************************************************************************/

void solve_ith_eqn_init (float *sum, float row[])
{
  *sum = row[matrixsize];
}

void solve_ith_eqn_g (int j, float *sum, float row[], float new)
{
  *sum = *sum + (-new*(row[matrixsize-j]));
}

void solve_ith_eqn_h (int rowno, float *sum, float row[])
{
  *sum = *sum/row[rowno];
}


/*********************************************************************************************************************
Creates a pipeline with the processors - each processor passes on already received values
**********************************************************************************************************************/

void pipeline (void (*init) (float *, float *),void (*g) (int, float *, float *, float),void (*h) (int, float *, float *),float local_vals[], float res[])
{
  float tmp, sum;
  int startelt, endelt, no_seen, no_to_receive;
  float accum[matrixsize];
  MPI_Status status;
  int n = N;  
  int x, i, j;
  MPI_Comm_size (MPI_COMM_WORLD, &p);
  MPI_Comm_rank (MPI_COMM_WORLD, &pid);
  if (pid==(n-1)%p)
    {
      (*init)(&sum, &local_vals[((n-1)/p)*(n+1)]);
      (*h)(n-1, &sum, &local_vals[((n-1)/p)*(n+1)]);
      res[(n-1)/p] = sum;
      MPI_Send(&sum, 1, MPI_FLOAT, mod(pid-1,p), 0, MPI_COMM_WORLD);
      accum[0] = sum;
      no_seen = 1;
    }
  else
    no_seen = 0;
  if (n<p)
    {
    if (pid < n-1)
      startelt = 0;
    else
      startelt = -1;
    }
  else if (n%p == 0)
    {
      if (pid < (n-1)%p)
        startelt = n/p - 1;
      else
        startelt = n/p - 2;
    }
  else if (pid > n%p - 2)
    startelt = n/p - 1;
  else
    startelt = n/p;
  if (pid>0)
    endelt = 0;
  else
    endelt = 1;
  
  //doing computations in parallel

  #pragma omp parallel for  
  for (x=startelt; x>=endelt; x--)
  {
    (*init)(&sum, &local_vals[x*(n+1)]);  
    for (i=0; i<no_seen; i++)
        (*g)(i+1, &sum, &local_vals[x*(n+1)], accum[i]);
    if (no_seen==0)
        no_to_receive = (n-pid-1)%p;
    else
        no_to_receive = p-1;
    for (j=0; j<no_to_receive; j++)
    {
      MPI_Recv(&tmp, 1, MPI_FLOAT, (pid+1)%p, 0, MPI_COMM_WORLD, &status);
      if ((j==0) && ((no_seen>0)||((no_seen==0)&&(pid==n%p))))
      ;
      else
          MPI_Send (&tmp, 1, MPI_FLOAT, mod(pid-1,p), 0, MPI_COMM_WORLD);
      no_seen++;
      accum[no_seen-1] = tmp;
      (*g)(no_seen, &sum, &local_vals[x*(n+1)], tmp);
    }

     
      (*h)(pid+p*x, &sum, &local_vals[x*(n+1)]); 
      res[x] = sum;
      MPI_Send(&sum, 1, MPI_FLOAT, mod(pid-1,p), 0, MPI_COMM_WORLD);

      no_seen++;
      accum[no_seen-1] = sum;
  }
  if (pid==0)
  {   
    (*init)(&sum, &local_vals[0]);
    for (i=0; i<no_seen; i++)
    (*g)(i+1, &sum, &local_vals[0], accum[i]);
    if (no_seen==0)
      no_to_receive = (n-pid-1)%p;
    else 
      no_to_receive = p-1;
    for (j=0; j<no_to_receive; j++)
    {
      MPI_Recv(&tmp, 1, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, &status);
      no_seen++;
      (*g)(no_seen, &sum, &local_vals[0], tmp);
    }
    (*h)(0, &sum, &local_vals[0]);
    res[0] = sum;
  }
}


void makecyclic(float inarray[], float outarray[])
{
  int maxblocksize, counter,i,j,index;

  if (matrixsize%p == 0)
    maxblocksize = matrixsize/p;
  else
    maxblocksize = matrixsize/p+1;

  for (counter=0;counter<matrixsize;counter++)
    {
      i = counter%p;
      j = counter/p;
      if (i<=(matrixsize-1)%p)
  index = maxblocksize*i+j;
      else
  index = maxblocksize*((matrixsize-1)%p+1) + 
    (matrixsize/p)*(i - ((matrixsize-1)%p+1)) + j;
      outarray[index] = inarray[counter];
    }
}

void makecyclicmatrix(float inarray[], float outarray[])
{
  int maxblocksize, counter,i,j,k,index;
  float x;

  if (matrixsize%p == 0)
    maxblocksize = matrixsize/p;
  else
    maxblocksize = matrixsize/p+1;
  for (counter=0;counter<matrixsize;counter++)
    {
      i = counter%p;
      j = counter/p;
      if (i<=(matrixsize-1)%p)
  index = maxblocksize*i+j;
      else
  index = maxblocksize*((matrixsize-1)%p+1) + 
    (matrixsize/p)*(i - ((matrixsize-1)%p+1)) + j;
      for(k=0;k<=matrixsize;k++)
  outarray[index*(matrixsize+1)+k] = inarray[counter*(matrixsize+1)+k];
    }
}

int mod(int x, int y)
{
  while (x<0)
    x=x+y;
  
  x = x%y;
  return x;
}

/*********************************************************************************************************************
It initializes the matrix A and Vector B with the computations
**********************************************************************************************************************/


void initialise_the_matrices()
{
  int i,j,n=N;
   
  if(pid==0)
  {
    for(i=1; i<=n; i++)
    {
      for(j=1; j<=n; j++)
      {
        if(i!=j)
              A_Orig[i][j]=((float)rand())/RAND_MAX;
       else
            A_Orig[i][j]=0.0;
      }
    }
    float partialSum;
    for(i=1; i<=n; i++)
    {
        partialSum=0;
        for(j=1; j<=n; j++)
        {
            partialSum = partialSum +  A_Orig[i][j];
        }
      for(j=i; j<=i; j++)
        {
      if(i==j)
          A_Orig[i][j]=1+partialSum;
        }
    }
    for(i=1; i<=n; i++)
    {
        X_Original[i]=rand() % 10 + 1;    
    }
  for(i=1; i<=n; i++)
   {
    mvSum=0.0;
    for(j=1; j<=n; j++)
    {
      mvSum+= A_Orig[i][j]*X_Original[j];
    }
    B1_Orig[i]=mvSum;

  }
 }    
}

