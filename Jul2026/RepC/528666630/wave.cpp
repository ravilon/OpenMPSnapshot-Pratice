/* CSCI 6330-- Parallel Processing Concepts
 * Class Example-- 1-D Wave Equation
 * Prof: Dr. K. Poudel
 * Group member: J. Gailbreath, J. Long. R. Morse, D. Shamsieva, E. White
 * Program:  This program implements a concurrent wave equation as described in 
 * Chpt. 5 of Fox et al., 1988, Solving Problems on Concurrent Processors, vol. 1.
 * We use OpenMP to implement parallel processing to solve the equation.
 *
 * This code is a derivative of one used with MPI as written by Saleh Elmohamed  11/22/97
*/

#include <iostream>	//	Several standard stream objects
#include <cstdio>	//	C-style input-output functions
#include <cstdlib>	//	General purpose utilities: program control, dynamic memory allocation, random numbers, sort and search
#include <cmath>	//	Common mathematics functions
#include <chrono>	//	C++ time utilites
#include <omp.h>	//	The OpenMP API specification for parallel programming
#include <algorithm>
#include <array>
#include <sys/time.h>


using namespace std;
//used for time tracking purposes
using namespace std::chrono;

//set a base number to represent pi
#define PI 3.14159

//establish variables used throughout the program
int nsteps = 100;                   	// number of time steps
int tpoints = 100;                  	// total points along string

//arrays for storing the values of amplitude on a "string" for discrete times
double values[102];             		// values at time t
double oldval[102];                   // values at time (t-dt)
double newval[102];                   // values at time (t+dt)
double rc = 0;								// generic return value for function

//simple function to determine initial values based on sine curve
double function(double x){
    double r;
    r = sin(2.0 * PI * x);
    return r;
}

//function to calculate the new values with the wave equation
double wave_function(int i){
    const double dtime = 0.3;
    const double c = 1.0;
    const double dx = 1.0;
    double tau, sqtau;
	double rc;
    tau = (c * dtime / dx);
    sqtau = tau * tau;
	//1-D wave equation
    rc = (2.0 * values[i]) - oldval[i] + (sqtau * (values[i - 1] - (2.0 * values[i]) + values[i + 1]));
	return rc;
}


int main() {

    // start the timer
	auto start = high_resolution_clock::now();

		for (int i = 1; i <= tpoints+1; i++) {
              int t = omp_get_thread_num();
              double x = ((double)(i % 10) / (double)(tpoints - 1));
              values[i] = function(x);
              oldval[i] = values[i];
              if(i < 10 || i > 90){
					  printf("thread#= %d  newval[%d]= %lf\n", t, i, values[i]);
			  }
          }

    printf("\n*****\nNOW FOR THE UPDATES\n*****\n");

		for (int i = 1; i < tpoints+1; i++) {
             int t = omp_get_thread_num();
             rc = wave_function(i);
			 newval[i]= rc;
             oldval[i] = values[i];
             values[i] = newval[i];
              if(i < 10 || i > 90){
			 		printf("thread#= %d  newval[%d]= %lf\n", t, i, values[i]);
			 }
     	}

    // Get ending timepoint
    auto stop = high_resolution_clock::now();

    // Get duration. Substart timepoints to
    // get duration. To cast it to proper unit
    // use duration cast method
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Time taken using serial: " << duration.count() << " microseconds" << endl;
/**************************************************************************************/

    // start the timer
	auto begin = high_resolution_clock::now();

	//establish parallel region to setup initial values for the string
	//this will display the current thread that is active and the value for the string that is produced
	int i;
#pragma omp parallel for  schedule(static) shared(values, oldval, newval, tpoints) private(i)
		for (i = 0; i <= tpoints+1; i++) {
			int t = omp_get_thread_num();
            double x = ((double)(i % 10) / (double)(tpoints - 1));
            values[i] = function(x);
            oldval[i] = values[i];
              if(i < 10 || i > 90){
					  printf("thread#= %d  newval[%d]= %.8lf\n", t, i, values[i]);
			  }
        }

    printf("\n*****\nNOW FOR THE UPDATES\n*****\n");
	//establish parallel region to update the amplitude
#pragma omp parallel for schedule(guided, 1)  shared(values, oldval, tpoints) private(newval, i, rc)
	for (i = 1; i <= tpoints+1; i++) {
			int t = omp_get_thread_num();
			#pragma omp critical
			{
					rc = wave_function(i);
					newval[i] = rc;
					oldval[i] = values[i];
        			values[i] = newval[i];
			}
              if(i < 10 || i > 90){
					printf("thread#= %d  newval[%d]= %.8lf\n", t, i, values[i]);
			}
	}
        
    // Get ending timepoint
    auto end = high_resolution_clock::now();

    // Get duration. Substart timepoints to
    // get duration. To cast it to proper unit
    // use duration cast method
    auto time = duration_cast<microseconds>(begin - end);

    cout << "Time taken using openmp: " << duration.count() << " microseconds" << endl;

    return 0;

}
