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
int LUPDecompose(double *A, int N, double Tol, int *P)
{

    int i,z, j, k, imax,flag=1;
    double maxA, absA;
            for (i = 0; i <= N; i++)
                P[i] = i;
            for (i = 0; i < N; i++)
            {
                maxA = 0.0;
                imax = i;
                for (k = i; k < N; k++)
                    if ((absA = fabs(A[k * N + i])) > maxA)     //Selecting the Pivot of the Column
                    {
                        maxA = absA;
                        imax = k;
                    }


                if (maxA > Tol)
                {
                    flag = 1;

                    if (imax != i)
                    {
                        //Pivoting vector P
                        j = P[i];
                        P[i] = P[imax];
                        P[imax] = j;

                        //Pivoting Rows of A
                        for (z = 0; z < N; z++)
                        {
                            A[(2 * N * N)] = A[i * N + z];
                            A[i * N + z] = A[imax * N + z];
                            A[imax * N + z] = A[(2 * N * N)];
                        }

                        P[N]++;
                    }

                    //Compute the Matrix A
                    for (j = i + 1; j < N; j++)
                    {
                        A[j * N + i] /= A[i * N + i];
                        for (k = i + 1; k < N; k++)
                        {
                            A[j * N + k] -= A[j * N + i] * A[i * N + k];
                        }
                    }
                }
                else
                    {
                        flag = 0; //Error in decomposition
                    }
            }
    return flag;
}


/*
 *
 * This function computes the inversion of the previous matrix A
 * The result is the the second part of the matrix A, starting from N*N.
 * This version is the task version, the matrix is quite regular, so adding task cause only overhead (see the version with omp for, is faster and don't use task, simpler)
 */
void LUPInvert(double *A, int *P, int N,int nthreads,int chunk)
{
    #pragma omp parallel shared(A,P,N,nthreads, chunk)
    {
        #pragma omp single
        {
            //is possible parallelize only the external loop because the inner region are critic
            #pragma omp task
            for (int j = 0; j < N; j++)
            {
                for (int i = 0; i < N; i++)
                {
                    if (P[i] == j)
                        A[(N*N)+i*N+j] = 1.0;
                    else
                        A[(N*N)+i*N+j] = 0.0;
                    //it's possible add a pragma omp but need at the end a taskwait and introduce delay (remove the // and the code is slower than the serial)
                    //#pragma omp task
                    for (int k = 0; k < i; k++)
                    {

                        A[(N*N)+i*N+j] -= A[i*N+k] * A[(N*N)+k*N+j]; // Computation of the inverse

                    }
                    //#pragma omp taskwait
                }
                #pragma omp task
                for (int i = N - 1; i >= 0; i--)
                {
                    #pragma omp task
                    for (int k = i + 1; k < N; k++)
                    {
                        A[(N * N) + i * N + j] -= A[i * N + k] * A[(N * N) + k * N + j];
                    }
                    //there is a loop carried dependence so wait to inner loop  end
                    #pragma omp taskwait

                    A[(N*N)+i*N+j] = A[(N*N)+i*N+j] / A[i*N+i];
                }
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
    double *A=calloc(1+(2*n*n),sizeof(double *)); //contiguous allocation
    // loop for fill the matrix
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
        {
            A[i*n+j]=rand()%10;
        }
    }


    int *P=malloc((n+1)*sizeof(int));
    double tol=0.000000000001; // to avoid the method degenerate
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
