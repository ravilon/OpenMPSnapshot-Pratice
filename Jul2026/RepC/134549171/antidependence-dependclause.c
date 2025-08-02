/*
 * In this example we show an anti-dependence expressed using the depend clause on the task construct.
 *
 * The program will always print " x = 1 ", because the depend clauses enforce the ordering of the tasks. If the depend
 * clauses had been omitted, then the tasks could execute in any order and the program would have a race condition.
*/

#include <stdio.h>
#include <omp.h>

int main()
{
   int x = 1;
   #pragma omp parallel
   #pragma omp single
   {
      #pragma omp task shared(x) depend(in: x)
         printf("x = %d\n", x);
      #pragma omp task shared(x) depend(out: x)
         x = 2;
   }
   return 0;
}














































