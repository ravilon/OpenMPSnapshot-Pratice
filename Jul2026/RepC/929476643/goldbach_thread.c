// Copyright 2022 Luis Solano <luis.solanosantamaria@ucr.ac.cr>

#include "goldbach_thread.h"

#include <assert.h>
#include <omp.h>
#include <stdio.h>

#include "data.h"
#include "goldbach_calculator.h"

/**
 @brief Crea y ejecuta los threads para hacer los cálculos
 @param shared_data Puntero al struct que contiene los datos compartidos
 @param numbers_count Cantidad de números que se van a trabajar 
 @param thread_count Cantidad de threads
 @see enum_error.h
*/
enum error_t goldbach_thread_run_omp(struct shared_data * shared_data,
  uint64_t numbers_count, uint64_t thread_count);


enum error_t goldbach_thread_setup_threads(uint64_t thread_count,
  struct numbers_array * numbers, struct bit_array * sieve) {
  assert(numbers);
  assert(sieve);
  // create shared_data for threads
  struct shared_data shared_data;
  // save numbers_array in shared_data
  shared_data.numbers = numbers;
  // save primes_array in shared_data
  shared_data.sieve = sieve;

  shared_data.thread_count = thread_count;
  shared_data.processed_numbers = 0;

  enum error_t error = ERROR_SUCCESS;

  error = goldbach_thread_run_omp(&shared_data, numbers->count, thread_count);

  return error;
}

enum error_t goldbach_thread_run_omp(struct shared_data * shared_data,
  uint64_t numbers_count, uint64_t thread_count) {
  assert(shared_data);
  enum error_t error = ERROR_SUCCESS;

  struct numbers_array * numbers = shared_data->numbers;
  assert(numbers);

  // for index := 0 to thread_count do
  # pragma omp parallel for num_threads(thread_count) \
      default(none) schedule(dynamic) \
      shared(numbers, numbers_count, shared_data)
  for (uint64_t index = 0; index < numbers_count; ++index) {
    // calculate_number(shared_data[my_index])
    goldbach_calculator_calculate_number(
      &(numbers->contained_numbers[index]), shared_data->sieve);
  }

  return error;
}
