#include <chrono>
#include <iostream>
#include <omp.h>
using namespace std;

/*
    f(x) = 4/(1 + x^2)
    area under this curve from 0 to 1 interval.
*/
long numSteps = 100000;
double deltaX;

void computePiSerial() {
  double x, pi = 0.0;
  deltaX = 1.0 / (double(numSteps));
  for (int i = 0; i < numSteps; i++) {
    x = (i + 0.5) * (deltaX);
    double fx = (4.0 / (1 + x * x)) * deltaX;
    pi = pi + fx;
  }
  cout << pi;
}

void computePiParallel() {
  // This implementation can be optimized using 2d padding for avoiding chaos
  // caused in cache access, and it slows down significantly, this 1d array can
  // be modelled using 2d array with coloumn 0 as array elements and rest as 0.
  // providing an uniform cache access pattern.
  const int MAX_NUM_T = 100;
  int numThreads;
  double pi = 0.0;
  double sum[MAX_NUM_T];
  deltaX = 1.0 / (double(numSteps));
#pragma omp parallel
  {
    double x;
    int id = omp_get_thread_num();
    int nThreads = omp_get_num_threads();
    // let only one thread record this thread count. To prevent all threads
    // writing to it.
    if (id == 0)
      numThreads = nThreads;
    for (int i = id; i < numSteps; i += nThreads) {
      x = (i + 0.5) * deltaX;
      double fx = (4.0 / (1.0 + x * x)) * deltaX;
      sum[id] += fx;
    }
  }
  // This serial loop is a performance bottleneck, can we elimiate it?
  for (int i = 0; i < numThreads; i++)
    pi += sum[i];
  cout << pi;
}

// Using Thread syncronization
void computePiParallelCriticalSection() {
  const int MAX_NUM_T = 100;
  int numThreads;
  double pi = 0.0;
  deltaX = 1.0 / (double(numSteps));
#pragma omp parallel
  {
    double x;
    int id = omp_get_thread_num();
    int nThreads = omp_get_num_threads();
    double localSum = 0.0;
    // let only one thread record this thread count. To prevent all threads
    // writing to it.
    if (id == 0)
      numThreads = nThreads;
    for (int i = id; i < numSteps; i += nThreads) {
      x = (i + 0.5) * deltaX;
      double fx = (4.0 / (1.0 + x * x)) * deltaX;
      localSum += fx;
    }
    // atomic is silightly faster than critical section based thread sync, as most
    // architectures have it in ISA itself, if no target specific atomic inst is
    // available in ISA, then compiler will fall back to generating code based on
    // critical section. The reason this code in not within for loop is due to the
    // fact that atomics or cs based thread sync introduces serial execution to the
    // code, and it will have a performance penalty for each loop iteration, in some
    // cases we might not use omp as well, that being said we can use a thread local
    // variable and finally after looping is done in data space by multiple threads,
    // it can be consolidated, assuming the opreation is associative, like sums and
    // products.
    #pragma omp atomic // or pragma omp critial
    pi += localSum;
  }
  // This serial loop is a performance bottleneck, can we elimiate it?
  cout << pi;
}

int main() {
  auto start_time = std::chrono::high_resolution_clock::now();
  computePiParallelCriticalSection();
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
      end_time - start_time);
  std::cout << "Time taken by function f: " << duration.count()
            << " nanoseconds" << std::endl;
  return 0;
}