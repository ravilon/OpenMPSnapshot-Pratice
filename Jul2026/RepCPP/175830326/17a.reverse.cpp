#include <cstdio>
#include <iostream>
#include <omp.h>

int main(int argc, char* argv[])
{
  #pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int n = omp_get_thread_num();
    for (int i = nthreads; i > 0; --i) {
      #pragma omp barrier
      {
        if (i == n + 1) {
          #pragma omp critical
          std::printf("Thread %d, iteration %d\n", n, i);
        }
      }
    }
  }
  std::cout << "Execution finished" << std::endl;

  return 0;
} 