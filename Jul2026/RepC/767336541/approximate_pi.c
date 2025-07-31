#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>

#define N 10000000
int inside_point = 0;

int main(){
    time_t t;
    srand((unsigned) time(&t));

    #pragma omp parallel for
    for (int i=0; i<N; i++){
        double x = (double)rand()/RAND_MAX;
        double y = (double)rand()/RAND_MAX;

        if (x*x + y*y <= 1.0){
            #pragma omp atomic update
            inside_point++;
        }
    }

    double approx_pi = 4.0*inside_point / N;
    printf("Approximation of pi = %f \n", approx_pi);

    return 0;
}