#include "first-primes.hpp"

void FirstPrimes::serial() {
u_int32_t p = 0, tmp, i, n;
for (n = 2; p < numPrimes; n++) {
isComposite = false;  // reset flag

for (i = 0; i < p; i++) {
tmp = composites[i];  // read to reduce memory access
if (tmp < n) tmp += primes[i];
if (tmp == n) isComposite = true;
composites[i] = tmp;  // write back
}

if (!isComposite) {
primes[p] = n;
composites[p] = n;
p++;
}
}
}

void FirstPrimes::parallel() {
u_int32_t p = 0, tmp, i, n;
for (n = 2; p < numPrimes; n++) {
isComposite = false;  // reset flag

#pragma omp parallel for private(tmp)
for (i = 0; i < p; i++) {
tmp = composites[i];  // read to reduce memory access
if (tmp < n) tmp += primes[i];
if (tmp == n) isComposite = true;
composites[i] = tmp;  // write back
}

if (!isComposite) {
primes[p] = n;
composites[p] = n;
p++;
}
}
}

void FirstPrimes::printParameters(runtype_e runType) {
if (runType == SERIAL) {
printf("Finding first %d primes serially.\n", numPrimes);
} else {
printf("Finding first %d primes in parallel, %d threads max.\n", numPrimes, omp_get_max_threads());
}
}

FirstPrimes::FirstPrimes(int numPrimes) {
this->numPrimes = numPrimes;
this->primes = new u_int32_t[numPrimes];
this->composites = new u_int32_t[numPrimes];
}

FirstPrimes::~FirstPrimes() {
delete this->primes;
delete this->composites;
}