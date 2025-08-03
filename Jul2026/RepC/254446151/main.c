// Approximating Pi with Monte Carlo methods.
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h> // time()
#include <math.h> // sqrt()


void show_help() {
printf("Monte Carlo Approximation of Pi.\n");
printf("Usage:\n");
printf("  ./montecarlo NUM_TRIALS [SEED]\n");
}


int main(int argc, char *argv[]) {
int i, num_trials, count, seed;
double x, y, z, pi;
struct drand48_data buffer;

if (argc < 2) {
show_help();
exit(1);
}

num_trials = atoi(argv[1]);
if (argc < 3) {
seed = time(NULL);
} else {
seed = atoi(argv[2]);
}

count = 0;

// Initialize RNG.
srand48_r(seed, &buffer);

#pragma omp parallel for
for (i = 0; i < num_trials; i++) {
// Pick a random point in the unit square.
drand48_r(&buffer, &x);
drand48_r(&buffer, &y);

// Compute the distance from the origin.
z = sqrt((x * x) + (y * y));

// If we're inside the unit circle, increment the counter.
if(z <= 1) {
#pragma omp critical
count += 1;
}
}

pi = ((double)count) / num_trials * 4;

printf("Approximated value of Pi = %g (%d trials)\n", pi, num_trials);

return 0;
}
