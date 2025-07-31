#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "../timer.h"

// LinearSystem structure
struct LinearSystem
{
    int n, m;                           // Size of the linear system. || n = rows && m = columns ||
    double **A;                         // Original coefficient matrix
    double **U;                         // Upper-triangular coefficient matrix
    double *b;                          // Right-hand side after the triangularization of the matrix 
    double *b_ori;                      // Original right-side vector
    double *x;                          // Solution vector of the system
}typedef LinearSystem;

/*------------------------------------------------------------------
 * Function:  error
 * Purpose:   Print an error message
 * In arg:    The message
 */

void error (const char *msg)
{
    printf("%s\n",msg);
    exit(EXIT_FAILURE);
}

/*------------------------------------------------------------------
 * Function:  printMatrix
 * Purpose:   Print a n x m matrix
 * In arg:    The name of the matrix, the data and the size
 */
void printMatrix (const char name[], double **A, int n, int m)
{   
    printf("===== %s =====\n",name);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
            printf("%lf ",A[i][j]);
        printf("\n");
    }
    printf("\n");
}

/*------------------------------------------------------------------
 * Function:  printVector
 * Purpose:   Print a n size vector
 * In arg:    The name, the vector and the size
 */
void printVector (const char name[], double *b, int n)
{   
    printf("===== %s =====\n",name);
    for (int i = 0; i < n; i++)
        printf("%lf\n",b[i]);
    printf("\n");
}

/*------------------------------------------------------------------
 * Function:  newLinearSystem
 * Purpose:   Build the LinearSystem structure from the input
 * In arg:    The input arguments from command line
 */
LinearSystem* newLinearSystem (int argc, char *argv[])
{
    int i, j, k, p;
    double v;
    char *buffer = (char*)malloc(sizeof(char)*50);
    size_t bufsize = 50;
    size_t chars;
    FILE *in;
    
    /* Allocate memory */
    LinearSystem *ls = (LinearSystem*)malloc(sizeof(LinearSystem));

    /* Read the matrix */
    in = fopen(argv[2],"r");
    chars = getline(&buffer,&bufsize,in);
    if (chars <= 0) error("Reading file");
    if (!fscanf(in,"%d %d %d",&ls->n,&ls->m,&k)) error("Reading file");

    /* Allocate continous block of memory for the matrix */
    ls->A = (double**)malloc(sizeof(double*)*ls->n);
    ls->A[0] = (double*)calloc(ls->n*ls->m,sizeof(double));
    ls->U = (double**)malloc(sizeof(double*)*ls->n);
    ls->U[0] = (double*)calloc(ls->n*ls->m,sizeof(double));
    for (i = 1; i < ls->n; i++)
    {   
        ls->A[i] = ls->A[0] + i*ls->m;
        ls->U[i] = ls->U[0] + i*ls->m;
    } 

    /* Read the non-zero values */
    for (p = 0; p < k; p++)
    {
        if (!fscanf(in,"%d %d %lf",&i,&j,&v)) error("Reading file");
        i--; j--;
        ls->A[i][j] = v;
        ls->U[i][j] = v;
    }
    fclose(in);

    /* Read the RHS */
    in = fopen(argv[3],"r");
    chars = getline(&buffer,&bufsize,in);
    if (chars <= 0) error("Reading file");
    if (!fscanf(in,"%d %d",&ls->n,&k)) error("Reading file");

    /* Allocate continous block of memory for the RHS */
    ls->b = (double*)calloc(ls->n,sizeof(double));
    ls->b_ori = (double*)calloc(ls->n,sizeof(double));

    /* Read the non-zero values */
    for (i = 0; i < ls->n; i++)
    {
        if (!fscanf(in,"%lf",&v)) error("Reading file");
        ls->b[i] = v; ls->b_ori[i] = v;
    }

    /* Allocate continous block of memory for the solution x */
    ls->x = (double*)calloc(ls->m,sizeof(double));

    //printMatrix("Matrix A",ls->A,ls->n,ls->m);
    //printVector("Vector b",ls->b,ls->n);

    fclose(in);
    free(buffer);

    return ls;
}

/*------------------------------------------------------------------
 * Function:  BackSubstitution_Column
 * Purpose:   Solve a linear system by back substitution using column-oriented algorithm
 * In arg:    Matrix A, right-handed side vector b, answer vector x and the size of the system
 * Parallel:  The first loop that copies the b array to x can be parallelized as well
              The outer loop cannot be parallelized because there is a dependency between the column
              The inner loop can be parallelized without a critical section
 */
void BackSubstitution_Column (double **A, double *b, double *x, int n)
{
    int row, col;

    #pragma omp parallel for
    for (row = 0; row < n; row++)
        x[row] = b[row];

    for (col = n-1; col >= 0; col--)
    {
        x[col] /= A[col][col];

        #pragma omp parallel for
        for (row = 0; row < col; row++)
            x[row] -= A[row][col]*x[col];
    }
}

/*------------------------------------------------------------------
 * Function:  swap_rows
 * Purpose:   Swap two rows of the linear system
 * In arg:    The matrix, the RHS and the indexes of the two rows
 */
void swap_rows (double **U, double *b, int i, int j)
{
    /* Pointers swap */
    double *aux = U[i];
    U[i] = U[j];
    U[j] = aux;
    
    double aux2;
    aux2 = b[i];
    b[i] = b[j];
    b[j] = aux2;
}

/*------------------------------------------------------------------
 * Function:  max_col
 * Purpose:   Return the row index of the maximum value of the column k
 * In arg:    The matrix, the column index and the size
 */
int max_col (double **U, int k, int n)
{
    int maxIndex = k;
    double maxValue = U[k][k];
    for (int i = k; i < n; i++)
    {
        if (U[i][k] > maxValue)
        {
            maxValue = U[i][k];
            maxIndex = i;
        }
    }
    return maxIndex;
}

/*------------------------------------------------------------------
 * Function:  forwardElimination
 * Purpose:   Serial forward elimination
 * In arg:    The matrix, the RHS and the size of the system
 */
void forwardElimination (double **U, double *b, int n, int m)
{
    int r;
    double *l = (double*)calloc(n,sizeof(double));
    
    // For every column
    for (int k = 0; k < n-1; k++)
    {
        r = max_col(U,k,n);
        if (r != k) swap_rows(U,b,k,r);
        // For every row
        #pragma omp parallel for
        for (int i = k+1; i < n; i++)
        {
            l[i] = U[i][k] / U[k][k];
            // Apply row operation to eliminate the elements below the pivot 
            for (int j = k+1; j < m; j++)
                U[i][j] = U[i][j] - l[i]*U[k][j];
            b[i] = b[i] - l[i]*b[k];
        }
    }

    free(l);
    //printMatrix("Matrix U",U,n,m);
    //printVector("Vector b",b,n);
}

/*------------------------------------------------------------------
 * Function:  checkSystem
 * Purpose:   Check if the solution of system is correct
 * In arg:    The original matrix of the system, the RHS, the solutio vector
 *           and the size of the system
 */
int checkSystem (double **A, double *x, double *b, int n, int m)
{
    double error = 0.0;
    for (int i = 0; i < n; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < m; j++)
            sum += A[i][j]*x[j];
        error += pow(b[i]-sum,2);
    }
    printf("Error = %.10lf\n",sqrt(error));
    if (error > 1.0e-03) return 0;
    else                 return 1;
}

/*------------------------------------------------------------------
 * Function:  freeLinearSystem
 * Purpose:   Free memory for the LinearSystem structure
 * In arg:    The structure
 */
void freeLinearSystem (LinearSystem *ls)
{
    free(ls->A[0]);
    free(ls->U[0]);
    free(ls->b);
    free(ls->b_ori);
    free(ls->x);
    free(ls);
}

/*------------------------------------------------------------------
 * Function:  Usage
 * Purpose:   Prints a message for the usage of the program
 * In arg:    The program name
 */
void Usage (const char program_name[])
{
    printf("========================== PARALLEL GAUSS ELIMINATION ==========================\n");
    printf("Usage:> %s <num_threads> <coef_matrix> <rhs_vector>\n",program_name);
    printf("<coef_matrix> = Coeffient matrix\n");
    printf("<rhs_vector> = Right-hand side vector\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("** The input files must be in the MatrixMarket format, '.mtx'. **\n");
    printf("** More info: http://math.nist.gov/MatrixMarket/ **\n");
    printf("================================================================================\n");
}

/*------------------------------------------------------------------ */
int main (int argc, char *argv[])
{
    int num_threads;
	double start, finish, elapsed;

    if (argc-1 < 3)
    {
        Usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    /* Get number of threads */
    num_threads = atoi(argv[1]);

    /* Build the LinearSystem structure */
    LinearSystem *ls = newLinearSystem(argc,argv);

    /* Serial Forward Elimination */
    /* Set the number of threads */
    omp_set_num_threads(num_threads);

    GET_TIME(start);
    forwardElimination(ls->U,ls->b,ls->n,ls->m);
    BackSubstitution_Column(ls->U,ls->b,ls->x,ls->n);
    //printVector("Solution x",ls->x,ls->n);
    GET_TIME(finish);

    if (checkSystem(ls->A,ls->x,ls->b_ori,ls->n,ls->m)) printf("[+] The solution is correct !\n");
    else                                                printf("[-] The solution is NOT correct !\n");
    
    elapsed = finish - start;
    printf("Time elapsed = %.10lf s\n",elapsed);
    
    freeLinearSystem(ls);

	return 0;
}
