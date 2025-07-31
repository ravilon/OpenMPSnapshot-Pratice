#include <omp.h>
#include <iostream>
#define NUM_THREADS 8

static long num_steps = 100000000;
double step;



int main(){
    
    int n_threads_heap;
    
    // False sharing happens because we created sum as array previously to preven to this here we used singl variable
    // to prevent Race condtion we use syncing.
    double pi=0, sum {0.0};
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
        double local_sum = 0.0;
        for(int i = thread_id; i<num_steps; i+=n_threads_stack){
            double x = (i-0.5)*step;

            local_sum += 4.0/(1.0+x*x);
        }


        // Reduction using Atomic Sync to prevent Race Condition(each thread step on each other collisons
        // Takes Advantage of hardware constructs to perfom operation seamlessly if possible otherwise does critical sync.
        // Reduction 
        #pragma omp atomic
        sum += local_sum;
    }

    run_time = omp_get_wtime() - start_time;

    pi = sum*step;

    std::cout<<"pi with "<<num_steps<<" steps is "<<pi<<" in "<<run_time<<" seconds\n";
    return 0;
}