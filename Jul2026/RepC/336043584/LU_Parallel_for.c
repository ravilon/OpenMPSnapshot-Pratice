#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<omp.h>
/*
 *
 * Decomposition Function, the result of this function is the Matrix A decomposed in the LU form with Pivoting.
 * A= L-E+U s.t. P*A=L*U
 * This function isn't parallelized because there are a lot of critical part and using the task or the critical region
 * insert only overhead because the code must be filled with taskwait and critical region.
 */
int LUPDecompose(double **A, int N, double Tol, int *P) {

    int i, j, k, imax; 
    double maxA, absA;
    int flag=1;
    for (i = 0; i <= N; i++)
        P[i] = i; 
    for (i = 0; i < N; i++) 
    {
        maxA = 0.0;
        imax = i;

        for (k = i; k < N; k++)
            if ((absA = fabs(A[k][i])) > maxA) 
            { 
          
                maxA = absA;
                imax = k;
                
            }

        if (maxA > Tol) 
        { 

        if (imax != i) 
        {
        	
         	j = P[i];
            P[i] = P[imax];
            P[imax] = j;
            
            //pivoting without loop (use row pointers)
            A[2*N+i] = A[i];
            A[i] = A[imax];
            A[imax] = A[2*N+i];
            P[N]++;
         
        }
        
        for (j = i + 1; j < N; j++) 
        {
            A[j][i] /= A[i][i];

            for (k = i + 1; k < N; k++)
                A[j][k] -= A[j][i] * A[i][k];
        }
        flag=1;
    	}
    	else
    	{
    		flag=0;
    	}
    }
    return flag;

}
/*
*
* This function compute the inverse, simpler is faster, parallelize the outer loop to have better performance
* i have tried to making a collapse but the results is the same.
*
**/
void LUPInvert(double **A, int *P, int N,int nthreads, int chunk) {
  	#pragma omp parallel shared(A,P,N,nthreads, chunk)
  	{
  	#pragma omp for schedule(dynamic,chunk)
    for (int j = 0; j < N; j++) {
    	
        for (int i = 0; i < N; i++) {
            if (P[i] == j) 
                A[N+i][j] = 1.0;
            else
                A[N+i][j] = 0.0;
            for (int k = 0; k < i; k++)
                A[N+i][j] -= A[i][k] * A[N+k][j]; //Computation of the result matrix
        }
        for (int i = N - 1; i >= 0; i--) {
            for (int k = i + 1; k < N; k++)
                A[N+i][j] -= A[i][k] * A[N+k][j];
            A[N+i][j] = A[N+i][j] / A[i][i];
        }
    }
    }
}
int main()
{
	
	int n;
	printf("Dim A: ");
	scanf("%d",&n);
	int chunk=10;
	int var;
	printf("Insert number of threads: ");
	scanf("%d",&var);
	omp_set_num_threads(var);
    int nthreads = omp_get_num_threads();
	double **A=malloc(3*n*sizeof(double *)); // in this version of the code the matrix the three matrix are contiguous but 
	//i have used a different type of pointer to avoid the loop of pivoting. i need to add a loop to make a pointer of rows.
	for (int i = 0; i < 2*n; i++) 
	{
  		A[i] = malloc(n*sizeof(double));
  
	}
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<n;j++)
		{
			A[i][j]=rand()%10;
		}
	}

	int *P=malloc((n+1)*sizeof(int));
	double tol=0.000000000001;
	 double begin,begin1;
    double end,end1;
    begin=begin1=omp_get_wtime();
    int flag = LUPDecompose(A,n,tol,P);
    end=end1=omp_get_wtime();
    printf("FunctionA: %lf\n",end1-begin1);
    if(flag!=0)
    {
        begin1 = omp_get_wtime();
        LUPInvert(A, P, n, nthreads, chunk);
        end1 = end = omp_get_wtime();
    }
    printf("FunctionB: %lf\n",end1-begin1);
    printf("LU: %lf\n",end-begin);

	
	
	
	
}
