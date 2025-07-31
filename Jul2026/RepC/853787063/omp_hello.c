#include <stdio.h>
#include <stdlib.h>
#include <omp.h> 

void hello(void);

int main(int argc, char* argv[]) {
    int thread_count = strtol(argv[1], NULL, 10);
    #pragma omp parallel num_threads(thread_count)
    {
        hello();
    }
}

void hello(void) {
    int my_rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();

    printf("Hello from thread %d of thread count %d\n", my_rank, thread_count);
}

// gcc -o omp_hello omp_hello.c -fopenmp