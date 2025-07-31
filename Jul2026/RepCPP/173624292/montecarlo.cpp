#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include <math.h>

double value_at(double x)
{
	// Declaring the function f(x) = sin(x) 
	return sin(x);
}

// Montecarlo Algorithm with openmp integrration
double montecarlo(double x0, double xn, double y0, double yn ,int n,int nthreads,double rectArea)
{
	
	double s = 0; // factor
    double x,y;
	int i;
	#pragma omp parallel for schedule(dynamic,nthreads)  default(none) private(i) shared(n,x,y,x0,y0,yn,xn,nthreads)  reduction(+:s)
	for (i = 0; i < n; i++) {
		//myid = omp_get_thread_num();
       
		x = x0 + (xn-x0)*((double) rand() / (RAND_MAX)); //Random integer generation bw [0,1]
		y = y0 + (yn-y0)*((double) rand() / (RAND_MAX));
	
		if (abs(y) <= abs(value_at(x)) ){
			//Check if area is above x = 0 for area = +ve area
			if (value_at(x)>0 && y >0 && y<=value_at(x) ){
					s =s+1;
			} 
			//Check if area is below x = 0 for area = -ve area
		    if(value_at(x) < 0 && y < 0 && y>=value_at(x) )	{
		    	s=s-1;
		    }
		}

	}
	
	return ((rectArea*s)/n);
}

// Main program  
int main()
{   

	// Range of definite integral 
	double x0 = 0;
	double xn = 3.14159;
    double x,y;
	// Number of grids.
	int n = 1500;
	int nthreads= 8;
	omp_set_num_threads(nthreads);
	
	double f = (xn-x0)/n;
	// FInding maximum and minimum
	double y0 = 0;
	double yn = 1;
	// Comment bock below can be used if  maximima and minima of functions are unknown for sin(x) bw [0,pi] max and min are [0,1] 
	/*#pragma omp parallel for schedule(dynamic,nthreads) 
	double y0 = 0;
	double yn = y0;
	for( int i = 0 ; i < n ; i++ ) {
			x = x0 + f*i;
			y = value_at(x);
			if(y < y0){
				y0=y;
			}
			if(y > yn){
				yn = y;
			}
	} */

	double rectArea = (xn-x0)*(yn-y0);
	printf("Value of integral is %f\n",
		(montecarlo(x0,xn,y0,yn,n,nthreads,rectArea)));
	
	return 0;
}