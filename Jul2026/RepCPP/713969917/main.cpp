#include <iostream>
#include <string>
#include <cstdint>
#include <md5.hpp>
#include <input.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef PART1
#define LEADING_ZEROS "00000"
#define NUM_LEADING_ZEROS 5
#endif
#ifdef PART2
#define LEADING_ZEROS "000000"
#define NUM_LEADING_ZEROS 6
#endif

size_t run(std::string input)
{
  uint32_t result = UINT32_MAX;

#ifdef _OPENMP
  std::cout << "OpenMP enabled with " << omp_get_max_threads() << " threads" << std::endl;

#pragma omp parallel
#endif
  {
#ifdef _OPENMP
    int id = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
#else
    int id = 0;
    int nthreads = 1;
#endif

    uint32_t i = id;
    std::string hash;

    while (true)
    {
      std::string concatened = input + std::to_string(i);
      hash = MD5(concatened).hexdigest();

      if (hash.substr(0, NUM_LEADING_ZEROS) == LEADING_ZEROS)
      {
#ifdef _OPENMP
#pragma omp critical
#endif
        {
          if (i < result)
          {
            result = i;
          }
        }
        break;
      }

      if (result != UINT32_MAX)
      {
        break;
      }

      i += nthreads;
    }
  }

  return result;
}

#ifndef TEST
int main(int argc, char const *argv[])
{
  Input input(__FILE__);

  std::cout << run(*input) << std::endl;

  return 0;
}
#endif