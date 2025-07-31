#include <omp.h>
#include <iostream>
#define NUM_THREADS 8

static long num_steps = 100000000;
double step;


int main(){
    
    int n_threads_heap;
    // double sum is promoted to double sum[num_threads] to prevent Race Condition(each thread step on each other collisons)
    double pi=0, sum[NUM_THREADS] = {0.0};
    double start_time, run_time;

    step = 1.0/(double) num_steps;

    
    start_time = omp_get_wtime();

    omp_set_num_threads(NUM_THREADS);

    //Split Loop Iteration Between Threads - Cyclic distribution of loop Index - Round robin distribution.
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int n_threads_stack = omp_get_num_threads();

        // OS will not always give the amount of threads we requested, sometimes it might give even less threads
        // so we get how many number of threads provided by the OS and only work on those threads
        if(thread_id == 0) {
            n_threads_heap = n_threads_stack;
        }

        // each thread is processing step with n_threads_heap alternative offsets
        // That is thread 3 process 3, 3+n_threads_heap, 3+2*n_threads_heap, 3+3*n_threads_heap
        for(int i = thread_id; i<num_steps; i+=n_threads_stack){
            double x = (i-0.5)*step;
            sum[thread_id] += 4.0/(1.0+x*x);
        }
    }

    run_time = omp_get_wtime() - start_time;

    //Reduction Operation
    for(int i=0; i<n_threads_heap; i++){
        pi += sum[i] * step;
    }

    std::cout<<"pi with "<<num_steps<<" steps is "<<pi<<" in "<<run_time<<" seconds\n";
    return 0;
}
