#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
const int count = 10000000;            ///< Number of array elements
const int threads = atoi(argv[1]);     ///< Number of parallel threads
const int random_seed = atoi(argv[2]); ///< RNG seed (Random Number Generator)

int *array = 0;                        ///< Target array
int max = -1;                          ///< Maximum element

double start_time, end_time;           ///< Starting and ending time points

/* Initialize RNG */
srand(random_seed);

/* Determine OpenMP support */
printf("====================================================\nOpenMP: %d\n", _OPENMP);

/* Generate random array */
array = (int *)malloc(count * sizeof(int));
for (int i = 0; i < count; i++) {
array[i] = rand();
}

/* Find maximum element */
#pragma omp parallel num_threads(threads) private(start_time, end_time) shared(array, count) reduction(max: max) default(none)
{
start_time = omp_get_wtime();
#pragma omp for
for (int i = 0; i < count; i++) {
if (array[i] > max) {
max = array[i];
}
}
end_time = omp_get_wtime();
printf("local maximum: %d || time elapsed: %0.7lf\n", max, end_time - start_time);
}

printf("total maximum: %d\n====================================================\n", max);

return 0;
}