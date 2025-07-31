#include <iostream>
#include <omp.h>
#define NUM_THREADS 8


static long num_steps = 100000000;
double step = 1.0/(double) num_steps;

int main(){
    double pi, sum=0.0, x;
    double start_time, run_time;

    omp_set_num_threads(8);

    start_time = omp_get_wtime();
    #pragma omp parallel for schedule(static) reduction(+:sum) private(x) shared(num_steps)
    {
        for(int i=1; i<=num_steps; i++){
            x = (i-0.5)*step;
            sum = sum + 4.0/(1.0+x*x);
        } 
    }

    run_time = omp_get_wtime() - start_time;
    pi = step * sum;

    std::cout<<"pi with "<<num_steps<<" steps is "<<pi<<" in "<<run_time<<" seconds\n";
    return 0;
}