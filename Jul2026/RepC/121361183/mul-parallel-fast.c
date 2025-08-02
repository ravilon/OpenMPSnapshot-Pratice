/*
	Execute the gauss elimination on square matrix with processors on striped partioning
*/
#include<solve-openmp.h>
/*
	data for solve in rowwise or columnwise partitioning
*/

int mul_striped_fast(long dim,int thread,double **mat,double **mat2)
{
	/*
		dim is dimension of data matrix
		thread is the numbers of threads
		mat is data matrix
	*/
	long i,j,k;
	/* Making alocations for data */
	#pragma omp for private(j)
	for(i=0;i<dim;i++)
	{
		for(j=0;j<dim;j++) mat2[i][j]=mat[i][j];
	}
	for(i=0;i<3;i++)
	{
		#pragma omp for private(k)
		for(j=0;j<dim;j++)
		{
			for(k=0;k<dim;k++) mat2[j][k]=mat2[j][k]*mat[j][k];
		}
	}
	return(0);
}
