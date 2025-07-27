#include "omp.h"
#include <stdio.h>
#define TMAX   16     	
#define ITERATIONS 4    

static long num_steps = 100000;
double step,sum;

void operation(int num_t)
{
	double x, pi,aux;
	sum = 0.0;
	int i;
	omp_set_num_threads(num_t);
	#pragma omp parallel reduction(+:sum)
	{
		#pragma omp for
		for (i=0; i<num_steps; i++) 
		{
			x = (i+0.5)*step;	
         		aux=4.0/(1.0+x*x);
			#pragma omp critical
         		sum = sum + aux;
		}
	}
	pi = step * sum;
}

int main () {
   double start; 
   double end; 
   start = omp_get_wtime(); 
   int avg_times[TMAX+1];
   step = 1.0/(double)num_steps;
   
   operation(1);
   
   for (int x = 0; x < ITERATIONS; ++x)
   {
      for (int i = 1; i <= TMAX; ++i)
      {
   	operation(i);
      }
   }
   printf("Sum is %f\n",sum);
   
	end = omp_get_wtime(); 
	printf("Work took %f seconds\n", end - start);


   return 0;
}
