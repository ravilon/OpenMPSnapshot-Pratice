#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <number_of_tosses>\n", argv[0]);
        return 1;
    }

    int no_of_tosses = strtol(argv[1], NULL, 10);
    int num_in_circle = 0;

    #pragma omp parallel reduction(+:num_in_circle) num_threads(4)
    {
        unsigned int seed = omp_get_thread_num();
        #pragma omp for
        for (int toss = 0; toss < no_of_tosses; toss++) {
            double x = (rand_r(&seed) / (double) RAND_MAX) * 2.0 - 1.0;
            double y = (rand_r(&seed) / (double) RAND_MAX) * 2.0 - 1.0;
            double distance_squared = x * x + y * y;
            if (distance_squared <= 1.0) {
                num_in_circle++;
            }
        }
    }

    double pi_estimate = 4.0 * num_in_circle / ((double) no_of_tosses);
    printf("Estimate of pi: %lf\n", pi_estimate);

    return 0;
}

// gcc -o omp_pi omp_pi.c -fopenmp