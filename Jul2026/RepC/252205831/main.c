#include <stdio.h>
#include <omp.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "libraries/c-vector/cvector.h"

#define CVECTOR_LOGARITHMIC_GROWTH

#define DEFAULT_N 100
#define OUTPUT_FILE_NAME "primes.txt"

// Find all primes from 1 to lastNumber inclusive
// Returns cvector with found primes
int* eratosthenesSieve(const int lastNumber);

int main(int argc, char **argv)
{
if (argc <= 1)
{
printf("Range boundary not provided\nUsing default value: %d\n", DEFAULT_N);
}
const int n = argc >= 2 ? atoi(argv[1]) : DEFAULT_N;

struct timeval stop, start;
gettimeofday(&start, NULL);

// Firstly find all primes in subset [1, ⌊√n⌋]
const int subsetSize = sqrt(n);
// printf("Subset size: %d\n", subsetSize);
int* primes = eratosthenesSieve(subsetSize);
const int subsetPrimesCount = cvector_size(primes);

int subsetPrimes[subsetPrimesCount]; // Can't use vector because address will change. Maybe should not use it at all
memcpy(subsetPrimes, cvector_begin(primes), subsetPrimesCount * sizeof(*primes));

#pragma omp parallel shared(subsetSize, primes, subsetPrimesCount, n)
#pragma omp for nowait
for (int potentialPrime = subsetSize+1; potentialPrime <= n; potentialPrime++)
{
bool isPrime = true;
for (int i = 0; i < subsetPrimesCount; i++)
{
if (potentialPrime % subsetPrimes[i] == 0)
{
isPrime = false;
break;
}
}

if (isPrime)
{
#pragma omp critical
cvector_push_back(primes, potentialPrime);
}
}

FILE* file;
if ((file = fopen(OUTPUT_FILE_NAME, "w")) == NULL)
{
fputs("Unable to open output file", stderr);
cvector_free(primes);
exit(EXIT_FAILURE);
}

for (int i = 0; i < cvector_size(primes); i++)
{
fprintf(file, "%d\n", primes[i]);
}

gettimeofday(&stop, NULL);
printf("Execution took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);

cvector_free(primes);
return 0;
}

int* eratosthenesSieve(const int lastNumber)
{
const int n = lastNumber;
cvector_vector_type(int) primes = NULL;

// Number is potenitial prime if it is true
// Numer is index+1
bool sieve[n];
memset(sieve, true, sizeof(sieve));

// One is not prime
sieve[0] = false;

int startingPoint = 1;
#pragma omp parallel shared(sieve, startingPoint, primes)
{
while (startingPoint < n)
{
// Find first prime and take it
int myPrime = 0;
#pragma omp critical
{
#pragma omp flush
if (startingPoint < n) // Safety check
{
for (; startingPoint < n; startingPoint++)
{
if (sieve[startingPoint]) // It's prime
{
myPrime = ++startingPoint;
cvector_push_back(primes, myPrime);
#ifdef DEBUG
int id = omp_get_thread_num();
printf("Thread %d found prime: %d\n", id, myPrime);
#endif
break;
}

}
}

}

if (myPrime == 0) // No prime found
break;

// Cross out multiples of selected prime
for (int i = myPrime*2-1; i < n; i+=myPrime)
sieve[i] = false;
}
}

return primes;
}
