/*
 * Architektury výpočetních systémů (AVS 2019)
 * Projekt c. 1 (ANN)
 * Login: xstupi00
 */

/**
 * @brief Returns output of the neuron as product of inputs, sums and bias
 * @param inputSize   - number of inputs the neuron
 * @param input       - pointer to neuron input array (identical for all neurons in the layer)
 * @param weights     - pointer to weights for all neurons in the layer
 * @param bias        - bias value of the neuron
 * @return Output of the neuron
 */
// #pragma omp declare simd uniform(inputSize, input) linear(weight:512) simdlen(8) notinbranch
// #pragma omp declare simd uniform(inputSize, input) linear(weight:784) simdlen(8) notinbranch
// #pragma omp declare simd uniform(inputSize, input) linear(weight) simdlen(8) notinbranch
float evalNeuron(
  size_t inputSize,
  const float* input,
  const float* weight,
  float bias
);
