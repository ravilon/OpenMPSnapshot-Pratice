#include <omp.h>
#include <iostream>
#include <cmath>
#define NUM_THREADS 8

static long num_steps = 100000000;
double step = 1.0/(double) num_steps;

int main(){
    int n_threads=0;
    double sum[NUM_THREADS] = {0.0}, pi=0.0, start_time, run_time;

    start_time = omp_get_wtime();
    omp_set_num_threads(NUM_THREADS);

    //Split Loop Iteration Between Threads - Offset Consequent distribution of loop Index
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();

        if(thread_id==0){
            n_threads = omp_get_num_threads();
        }

        int offset = std::ceil(num_steps/n_threads);

        for(int i=offset*thread_id;(i<num_steps and i<offset*(thread_id+1)); i++){
            double x = (i-0.5)*step;
            sum[thread_id] += 4.0/(1.0+x*x);
        }
    }

    run_time = omp_get_wtime() - start_time;

    //Reduction Operation
    for(int i=0; i<n_threads; i++){
        pi += sum[i] * step;
    }

    std::cout<<"pi with "<<num_steps<<" steps is "<<pi<<" in "<<run_time<<" seconds\n";
    return 0;
}
