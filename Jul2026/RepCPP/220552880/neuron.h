/*
 * Architektury výpočetních systémů (AVS 2019)
 * Projekt č. 1 (ANN)
 * Login: xharmi00
 */


#ifndef NEURON_H
#define NEURON_H


#include <cstdlib>


/**
 * @brief Returns output of the neuron as product of inputs, sums and bias.
 *
 * @param inputSize   - number of inputs of the neuron
 * @param input       - pointer to neuron input array (identical for all
 *                      neurons in the layer)
 * @param weights     - pointer to weights for the neuron
 * @param bias        - bias value of the neuron
 * @return Output of the neuron.
 */
// #pragma omp declare simd uniform(inputSize, input) linear(weights:784)
// #pragma omp declare simd uniform(inputSize, input) linear(weights:512)
float evalNeuron(
	size_t inputSize,
	const float *input,
	const float *weights,
	float bias
);


#endif
