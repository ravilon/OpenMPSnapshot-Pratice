/* calculatePI.cpp -- Vinicius Almada
 * The purpose of this program is to calculate 
 * PI number with multithread programming.
 * PI can be obtained from integral of 
 * 4 / (1 + x²) from 0 to 1.
 *
 * To obtain this value approximattely we can 
 * use sum of rectangles:
 * PI = sum(F(x_i) * dx) with i equals 0 to N.
 *
 * As larger N the PI will be better approximated.
 *
 * Approach with 1 bilion.
 */
#include <cstdio>
#include <cmath>
#include "omp.h"

/* Calculate function for a given x */
double integralFunction(double x);

/* Calculate PI for a given number of steps
 * and with a given number of threads. */
double calculatePI(double down, double up, long numSteps, int numberOfThreads = 1);

/* The block of code that will be used for each thread separately */
double parallelBlock(long newNumSteps, double threadDown, double threadUp);

int main() {

	printf("Running on a system with %d threads\n\n", omp_get_max_threads());

	int i = 1;
	while (true) {
		double pi = calculatePI(0.0, 1.0, 1'000'000'000L, i++);
		
		printf("pi = %.20f\n", pi);
		
		if (i == omp_get_max_threads() + 1)
			break;
	}
}

double integralFunction(double x) {
	return 4.0 / (1.0 + x * x);
}

double calculatePI(double down, double up, long numSteps, int numberOfThreads) {
	
	double sum = 0.0;
	double start = omp_get_wtime();
#pragma omp parallel num_threads(numberOfThreads)
	{
		double threadDown = (double) omp_get_thread_num() / (double) numberOfThreads;
		double threadUp = (double) (omp_get_thread_num() + 1) / (double) numberOfThreads;
		double temp = parallelBlock(numSteps / numberOfThreads, threadDown, threadUp);
#pragma omp critical
		sum += temp;
	}
	double end = omp_get_wtime();
	printf("%2d threads in %.8f seconds\t", numberOfThreads, end - start);
		
	return sum;
}

double parallelBlock(long newNumSteps, double threadDown, double threadUp) {
	double sum = 0.0;
	
	double dx = (threadUp - threadDown) / (double) newNumSteps;
	
	for (long i = 0; i < newNumSteps; i++) {
		double x = (i + 0.5) * dx;
		x += threadDown;
		double funRes = integralFunction(x);
		sum += funRes * dx;
	}
	return sum;
}
