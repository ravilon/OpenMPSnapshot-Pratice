
#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include <math.h>

double value_at(double x)
{
	// Declaring the function f(x) = sin(x) 
	return sin(x);
}

// Function to evalute the value of integral 
double trapezoidal(double a, double b, int n,int nthreads)
{
	// Grid spacing 
	double h = (b - a) / n;

	// Computing sum of first and last terms 
	// in above formula 
	double s = value_at(a) + value_at(b);

	int i;
	int myid ;
	#pragma omp parallel for schedule(dynamic,nthreads)  default(none) private(i) shared(n,a,h,nthreads)  reduction(+:s)
	for (i = 0; i < n; i++) {
		//myid = omp_get_thread_num();

		//printf("Section 2: From thd num %d out of %d thds : i = %d \n", myid, nthreads, i);
		s =s + 2 * value_at(a + i * h);
	}
	
	return (h / 2)*s;
}

// Main program 
int main()
{   

	// Range of definite integral 
	double x0 = 0;
	double xn = 3.14159;

	// Number of grids. 
	int n = 1500;
	int nthreads= 6;
	omp_set_num_threads(nthreads);
	printf("Value of integral is %f\n",
		(trapezoidal(x0, xn, n,nthreads)));
	return 0;
}
