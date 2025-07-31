#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    int inside_circle = 0;
    int total_points = 1000000; // Adjust total_points as needed
    int num_threads = 4; // Adjust the number of threads as needed
	printf("-----DHIVYESH R K---2021BCS0084\n");
    #pragma omp parallel num_threads(num_threads) reduction(+:inside_circle)
    {
        int local_inside_circle = 0;

        unsigned int seed = omp_get_thread_num();

        #pragma omp for
        for (int i = 0; i < total_points; i++) {
            double x = (double)rand_r(&seed) / RAND_MAX;
            double y = (double)rand_r(&seed) / RAND_MAX;

            if ((x * x + y * y) <= 1) {
                local_inside_circle++;
            }
        }

        inside_circle += local_inside_circle;
    }

    double pi_estimate = 4.0 * inside_circle / total_points;

    printf("Estimated value of pi (parallel version with reduction): %f\n", pi_estimate);
    printf("Classical value of pi: %f\n", 3.14159265358979323846);

    return 0;
}

