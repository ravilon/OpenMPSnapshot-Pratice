// Author:        Rachna Sidana
// Description:   This program applies Gaussian Elimination to find the solution of Linear Systems of Equations. It also
//                calculates the execution time of code that has been parallelized.
//                OpenMp

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#define size  1000 // no. of columns 

int main()
{
    int i,j,k,n=size;
    float A[size][size],c,partialSum,X_Original[size],x[size], B[size], sum=0.0, mvSum=0.0;
    struct timeval start_time, stop_time, elapsed_time;  // timers
    double  numFlops;
    float gflops;
    
    //assigning values to matrix A
    
    for(i=1; i<=n; i++)
    {
        for(j=1; j<=n; j++)
        {
	        if(i!=j)
                A[i][j]=((float)rand())/RAND_MAX;
	       else
	            A[i][j]=0;
        }
    }
    
    //assigning values to matrix A's diagonal elements 
    // so that it is diagonally dominant

    for(i=1; i<=n; i++)
    {
        partialSum=0;
        for(j=1; j<=n; j++)
        {
            partialSum = partialSum +  A[i][j];
        }
	    for(j=i; j<=i; j++)
        {
		  if(i==j)
		      A[i][j]=1+partialSum;
        }
    }

    //assigning random values to vector X
   
    for(i=1; i<=n; i++)
    {
        X_Original[i]=rand() % 100 + 1;    
       
    }
   
    //Computing B

    for(i=1; i<=n; i++)
    {
        mvSum=0.0;
        for(j=1; j<=n; j++)
        {
            mvSum+=A[i][j]*X_Original[j];
        }
        B[i]=mvSum;
       
    }
    printf("\n*************************************************************");
    printf("\nApplying the algorithm using MatrixA and calculated VectorB \n");
    printf("***************************************************************\n");
   
    gettimeofday(&start_time,NULL);

   /* loop for the generation of upper triangular matrix*/

    for(j=1; j<=n; j++) 
    {
	#pragma omp parallel for private(i,k,c) 
        for(i=1; i<=n; i++)
        {
            if(i>j)
            {
                c=A[i][j]/A[j][j];
                for(k=1; k<=n+1; k++)
                {
                    if(k==n+1)
                        B[i]=B[i]-c*B[j];
                    else
                        A[i][k]=A[i][k]-c*A[j][k];
                }
            }
        }
    }
    x[n]=B[n]/A[n][n];

    /* this loop is for backward substitution*/

    for(i=n-1; i>=1; i--)
    {
        sum=0;
        for(j=i+1; j<=n; j++)
        {
            sum=sum+A[i][j]*x[j];
        }
        x[i]=(B[i]-sum)/A[i][i];
    }
    int N =n;
    gettimeofday(&stop_time,NULL);
    timersub(&stop_time, &start_time, &elapsed_time); // Unix time subtract routine
    printf("Total time was %f seconds.\n", elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0);
    numFlops = ((2.0f*N*N*N/3.0f)+(3.0f*N*N/2.0f)-(13.0f*N/6.0f));
    float flops = numFlops/(elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0);
    gflops = flops/1000000000.0f;
    printf("GFlops :  %f .\n",gflops); 
    int flag=1;   
    return(0);
}
