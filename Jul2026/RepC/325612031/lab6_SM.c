#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {

	int in_points = 0;
	int out_points = 0;

	int i;
	double x, y, z;

	#pragma omp parallel default(none)
	private(i, x, y) shared(in_points, out_points)
	#pragma omp for
   	for (i = 0; i < 10000000; i++) {
     	x = (rand()  % 100) / 100.0;
		y = (rand()  % 100) / 100.0;

      	if ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) <= 0.5 * 0.5) {
			in_points++;
		}
		else {
			out_points++;
		}
   	}

	double pi = 4.0 * in_points / (in_points + out_points);
	
	printf("%lf\n", pi);	
}
