/* File:    omp_trap3.c
 * Purpose: Estimate definite integral (or area under curve) using the
 *          trapezoidal rule.  This version uses a parallel for directive
 *
 * Input:   a, b, n
 * Output:  estimate of integral from a to b of f(x)
 *          using n trapezoids.
 *
 * Compile: gcc -g -Wall -fopenmp -o omp_trap3 omp_trap3.c
 * Usage:   ./omp_trap3 <number of threads>
 *
 * Notes:   
 *   1.  The function f(x) is hardwired.
 *   2.  In this version, it's not necessary for n to be
 *       evenly divisible by thread_count.
 *
 * IPP:  Section 5.5 (pp. 224 and ff.)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void Usage(char* prog_name);
double f(double x);    /* Function we're integrating */
double Trap(double a, double b, int n, int thread_count);

int main(int argc, char* argv[]) {
   double  global_result = 0.0;  /* Store result in global_result */
   double  a, b;                 /* Left and right endpoints      */
   int     n;                    /* Total number of trapezoids    */
   int     thread_count;

   if (argc != 5) Usage(argv[0]);
   thread_count = strtol(argv[1], NULL, 10);
   a = strtod(argv[2],NULL);
   b = strtod(argv[3],NULL);
   n = strtol(argv[4],NULL,10);

   global_result = Trap(a, b, n, thread_count);

   printf("With n = %d trapezoids, our estimate\n", n);
   printf("of the integral from %f to %f = %.14e\n",
      a, b, global_result);
   return 0;
}  /* main */

/*--------------------------------------------------------------------
 * Function:    Usage
 * Purpose:     Print command line for function and terminate
 * In arg:      prog_name
 */
void Usage(char* prog_name) {

   fprintf(stderr, "Usage: %s <number of threads> <a> <b> <n>\n", prog_name);
   exit(0);
}  /* Usage */

/*------------------------------------------------------------------
 * Function:    f
 * Purpose:     Compute value of function to be integrated
 * Input arg:   x
 * Return val:  f(x)
 */
double f(double x) {
   double return_val;

   return_val = x*x;
   return return_val;
}  /* f */

/*------------------------------------------------------------------
 * Function:    Trap
 * Purpose:     Use trapezoidal rule to estimate definite integral
 * Input args:  
 *    a: left endpoint
 *    b: right endpoint
 *    n: number of trapezoids
 * Return val:
 *    approx:  estimate of integral from a to b of f(x)
 */
double Trap(double a, double b, int n, int thread_count) {
   double  h, approx;
   int  i;
   int *iterations;

   iterations = (int*)malloc(sizeof(int)*n);
   h = (b-a)/n; 
   approx = (f(a) + f(b))/2.0; 
   iterations[0] = 0;
#  pragma omp parallel for num_threads(thread_count) \
      reduction(+: approx) schedule(runtime)
   for (i = 1; i < n; i++)
   {
     iterations[i] = omp_get_thread_num();
     approx += f(a + i*h);
   }
   iterations[n-1] = 0;
   approx = h*approx; 

   printf("\nPrinting array of iterations:\n");
   for (i = 1; i < n-1; i++)
   	printf("Iteration %d >> Thread %d\n", i, iterations[i]);
   printf("\n");

   free(iterations);

   return approx;
}  /* Trap */
