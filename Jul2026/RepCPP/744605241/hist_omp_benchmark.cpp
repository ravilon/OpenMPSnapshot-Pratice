#include "walltime.h"
#include <iostream>
#include <random>
#include <omp.h>
#include <fstream>

#define VEC_SIZE 1000000000
#define BINS 16
#define MAX_THREADS 16

using namespace std;

int main()
{
  // Initialize random number generator
  unsigned int seed = 123;
  float mean = BINS / 2.0;
  float sigma = BINS / 12.0;
  std::default_random_engine generator(seed);
  std::normal_distribution<float> distribution(mean, sigma);

  // Generate random sequence
  int *vec = new int[VEC_SIZE];
  for (long i = 0; i < VEC_SIZE; ++i)
  {
    vec[i] = int(distribution(generator));
    if (vec[i] < 0)
      vec[i] = 0;
    if (vec[i] > BINS - 1)
      vec[i] = BINS - 1;
  }

  ofstream csvFile("execution_times.csv");
  csvFile << "Type_nthreads,Time" << endl;

  // Sequential histogram computation
  long sequential_dist[BINS] = {0};
  volatile double seq_time_start = wall_time(); // volatile to prevent compiler optimization (that prints 0 sec.)
  for (long i = 0; i < VEC_SIZE; ++i)
  {
    sequential_dist[vec[i]]++;
  }
  volatile double tot_seq_time = wall_time() - seq_time_start;
  
  csvFile << "SEQ_1," << tot_seq_time << endl;

  // Parallel histogram computation
  for (int num_threads = 1; num_threads <= MAX_THREADS; ++num_threads)
  {
    omp_set_num_threads(num_threads);

    // Initialize histogram
    long dist[BINS] = {0};

    double time_start = wall_time();

#pragma omp parallel
    {
      long local_dist[BINS] = {0};
#pragma omp for
      for (long i = 0; i < VEC_SIZE; ++i)
      {
        local_dist[vec[i]]++;
      }

#pragma omp critical
      {
        for (int i = 0; i < BINS; ++i)
        {
          dist[i] += local_dist[i];
        }
      }
    }

    double time_end = wall_time();
    csvFile << "OMP_" << num_threads << "," << (time_end - time_start) << endl;
  }

  delete[] vec;
  csvFile.close();

  return 0;
}
