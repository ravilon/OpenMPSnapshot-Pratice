#inlcude <stdio.h>
#inlcude <stdlib.h>
#include <omp.h>

int main(int argc, char*argv[]){

	int i;
	double x, pi, sum = 0.0;
	long numSteps = atol(argv[1]);
	double step = 1.0 / (double)numSteps;

#pragma omp parallel for private(x)
	for (i=0; i<numSteps; ++i){
		x = (i+0.5)*step;
		double aux = 4.0/(1.0+x*x);

#pragma omp critical
		sum += aux;
	}
	pi = step * sum;

	printf("Valor de pi: %f\n", pi);
	return 0;
}