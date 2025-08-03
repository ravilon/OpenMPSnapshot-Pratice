#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>

#define R 1.0

/**
*
* MPI + OpenMP version.
*
*/

double get_random() {
return (double) rand() / (double) RAND_MAX;
}

double mpi_omp_pi(int num_trials, int argc, char* argv[]) {

double pi, x, y;
int num_points_circle = 0;
int i, start_index, n, rank, size;

MPI_Init(&argc, &argv); // Start MPI.

MPI_Comm_size(MPI_COMM_WORLD, &size); // Get MPI size.
MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get MPI rank.
srand(time(NULL)^rank); // randomize seed with current rank.

const int chunk_size = (num_trials+size)/size;

if(size < 2) {
printf("This application is meant to be run with more than 2 MPI processes.\n");
MPI_Abort(MPI_COMM_WORLD, 0);
}

start_index = rank*chunk_size;
n = (rank+1)*chunk_size;
if(n > num_trials) n = num_trials;

if(rank == 0) { // Master process.
double start_time = omp_get_wtime();
int tmp;

#pragma omp parallel for private(i) firstprivate(x,y) reduction(+:num_points_circle)
for(i = start_index ; i < n; ++i) {
x = get_random();
y = get_random();
if(x*x+y*y <= R) num_points_circle++;
}

for(i = 0; i < size-1; ++i) { // Receive results from other MPI processes.
MPI_Recv(&tmp, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, 0);
num_points_circle += tmp;
}

pi = 4.0*(num_points_circle/(double) num_trials);

double time_taken = omp_get_wtime() - start_time;
printf("PI = %.8f -- Time taken in MPI-OMP = %.8fs.\n", pi, time_taken);
} else { // Slave processes.
#pragma omp parallel for private(i) firstprivate(x,y) reduction(+:num_points_circle)
for(i = start_index ; i < n; ++i) {
x = get_random();
y = get_random();
if(x*x+y*y <= R) num_points_circle++;
}

// Send results.
MPI_Send(&num_points_circle, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
}

MPI_Finalize(); // Close MPI.

return pi;
}

int main(int argc, char* argv[]) {
const int num_trials = atoi(argv[1]);
mpi_omp_pi(num_trials, argc, argv);

return 0;
}
