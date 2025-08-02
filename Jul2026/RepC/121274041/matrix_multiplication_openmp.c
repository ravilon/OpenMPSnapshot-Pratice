/*
 * Filename: matrix_multiplication_openmp.c
 * Author: Pradeep Singh
 * Date:  03/12/18
 * Description: Matrix Multiplication using OpenMP.
 *              The program computes the time required to multiply two matrices the exection with
 *              different number of threads, which are taken from the command line argument
 *
 *               o Program usage is as,
 *
 *                          - gcc -fopenmp mat.c -o mat                             // for compiling the program
 *                          - time ./mat   (thread count)                           // run with time function
 *                          - valgrind --tool=memcheck --leak-check=yes ./mat       // check for segmentation faults (memory leaks)
 *                          - gcc -Wall -Werror -fopenmp -o mat mat.c               // it will check if any warnings
 *
 *              o PBS Usage: - qsub -v T=2 batch.mat                                // submit job
 *                           - qstat                                                // check job status
 *                           - cat mat.ojob_id                                      // check o/p
 */

// Include the necessary header files
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#define m 2000
#define n 3000
#define o 2000

// declaring matrices of NxN size
double **mat1, **mat2, **result, **trans;
int thread_count;
int thread_num;
double sum;
// Main program starts here
int main(int argc, char *argv[])
{

   if (argc != 2) {
      fprintf( stderr, "%s <number of threads>\n", argv[0] );
      return -1;
   }
   
   thread_count = atoi( argv[1] );

   int i, j, k;
   struct timeval start, end;

   omp_set_dynamic(0);
   omp_set_num_threads(thread_count);
   thread_num = omp_get_max_threads ( );

  printf ( "\n" );
  printf ( "  The number of threads used    = %d\n", thread_num );

/* Dynamic memory allocation for Matrices*/
/* Allocate memory for matrix rows and columns = 1000 X 1000 */

/* Matrix -- 1 */
   mat1 = (double **) malloc(m * sizeof(double *));       /* allocating memory to rows */
   for (i=0;i<m;i++)                                      /* allocating memory to col */
       mat1[i] = (double *) malloc(n * sizeof(double));

/* Matrix -- 2 */
   mat2 = (double **) malloc(n * sizeof(double *));       /* allocating memory to rows */
   for (i=0;i<n;i++)                                      /* allocating memory to col */
       mat2[i] = (double *) malloc(o * sizeof(double));

/* Result Matrix */
   result = (double **) malloc(m * sizeof(double *));     /* allocating memory to rows */
   for (i=0;i<m;i++)                                      /* allocating memory to col */
       result[i] = (double *) malloc(o * sizeof(double));

/* transpose Matrix */
   trans = (double **) malloc(o * sizeof(double *));     /* allocating memory to rows */
   for (i=0;i<o;i++)                                     /* allocating memory to col */
       trans[i] = (double *) malloc(m * sizeof(double));

/* Generating matrix elements with random numbers between 0 and 1 */
   srand(time(NULL));                                 /* srand() sets the seed for rand() */
   for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++) {
         mat1[i][j] = (double) rand()/ RAND_MAX;      // generates random number between 0 and 1
      }
   }

   for (i = 0; i < n; i++) {
      for (j = 0; j < o; j++) {
         mat2[i][j] = (double) rand()/ RAND_MAX;       // generates random number between 0 and 1
      }
   }

   gettimeofday(&start, NULL);                         /* start measuring time */

#  pragma omp parallel  num_threads(thread_num) \
      reduction(+: sum) private(i,j,k) shared(mat1,mat2)
   {

   // matrix multiplication
   # pragma omp for
   for (i = 0; i < m; i++) {
       for (j = 0; j < o; j++) {
           result[i][j] = 0;
           sum = 0;
           for (k =0; k < n; k++) {
               sum+= mat1[i][k] * mat2[k][j];
           }
           result[i][j] =sum;
       }
   }

   //  transpose of the matrix
   # pragma omp for
   for (i = 0; i < m; i++) {
       for (j = 0; j < o; j++) {
           trans[j][i] = result[i][j];
       }
   }
}

   gettimeofday(&end, NULL);    /* function to measure execution time */

   /* Print the execution time */
   printf("\n Execution Time: %fs \n", ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1000000.0));

   // Free memory (deallocate the memory)
   /* Free the allocated memory for all three matrices using free() */
   for (i = 0; i < m; i++){
       free(mat1[i]);
   }
    free(mat1);

   for (i = 0; i < n; i++){
       free(mat2[i]);
   }
   free(mat2);

   for (i = 0; i < m; i++){
       free(result[i]);
   }
   free(result);

   for (i = 0; i < o; i++){
       free(trans[i]);
   }
   free(trans);

   return 0;
}
