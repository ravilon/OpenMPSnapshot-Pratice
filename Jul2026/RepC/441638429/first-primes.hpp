#pragma once

#include "general_utils.hpp"

class FirstPrimes : public Project {
private:
int numPrimes;
unsigned int* primes;
unsigned int* composites;
bool isComposite;

public:
FirstPrimes(int numPrimes);
~FirstPrimes();

/**
* Serial implementation of the sieve-based prime finding method.
*/
void serial();

/**
* Parallel implementation of the sieve-based prime finding method. The parallel part is on the composite number
* updates.
*/
void parallel();

/**
* Print information about the game.
*/
void printParameters(runtype_e runType);
};
