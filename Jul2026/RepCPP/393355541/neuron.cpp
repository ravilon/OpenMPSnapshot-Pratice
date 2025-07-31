/*
 * Architektury výpočetních systémů (AVS 2019)
 * Projekt c. 1 (ANN)
 * Login: xstupi00
 */

#include <cstdlib>
#include <cstdio>
#include <math.h>
#include "neuron.h"


float evalNeuron(
  size_t inputSize,
  const float* input,
  const float* weights,
  float bias
)
{
  //TODO: Step0 - Fill in the implementation, all the required arguments are passed.
  //              If you don't use them all you are doing something wrong!
  #pragma omp simd reduction(+:bias) linear(input, weights) simdlen(8) aligned(weights)
  for(size_t i = 0; i < inputSize; i++) {
    bias += input[i] * weights[i];
  }
  return fmax(0.0, bias);
}
