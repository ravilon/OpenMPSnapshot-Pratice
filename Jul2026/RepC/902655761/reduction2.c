#include <stdio.h>
#include <omp.h>
#include <math.h>

#define n 1000

int main(){
    int dot=0;
    float x[n];
    int y[n];
    int i, c;
    float d=0.0;

    for (i = 0; i < n; i++){
        x[i]=(float)i;
        y[i]=i*2+1;
    }
    c=y[0];

	#pragma omp parallel for private(i) shared(x, y) reduction(min:c) reduction(max:d)
	for (i=0; i<n; i++){
	     if (y[i] < c)
	         c = y[i];

	     d = fmaxf(d,x[i]);
	}

    printf("Maior de x: %f \n", d);
    printf("Menor de y: %d \n", c);

    return 0;
}