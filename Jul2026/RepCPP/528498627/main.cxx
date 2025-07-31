#include <cstdint>
#include <random>
#include <vector>
#include <string>
#include <atomic>
#include <cstdio>
#include <omp.h>
#include "src/main.hxx"

using namespace std;




template <class T, class V>
int threadRun(vector<T>& x, int L) {
  int a = 0;  // parity sum
  V def = V();
  random_device dev;
  default_random_engine rnd(dev());
  uniform_int_distribution<> idxDis(0, int(x.size()-1));
  for (int l=0; l<L; ++l) {
    int i = idxDis(rnd);
    int j = idxDis(rnd);
    x[i]  = randomWithParity1(def, rnd, 0);
    a    += parity1(x[j]);
  }
  return a;
}


template <class T, class V>
void runWithTypeAndSize(int size, int accesses, int numThreads, int repeat, const string& name) {
  vector<T>   sharedData(size);
  vector<int> paritySum(numThreads);
  float t = measureDuration([&]() {
    for (int i=0; i<size; ++i)
      sharedData[i] = V();
    #pragma omp parallel for schedule(static, 1)
    for (int t=0; t<numThreads; ++t)
      paritySum[t] = threadRun<T, V>(sharedData, accesses);
  }, repeat);
  float totalParitySum = float(sumValues(paritySum)) / repeat;
  printf("[%09.3f ms; %.1f par_sum] %s {size=%d}\n", t, totalParitySum, name.c_str(), size);
}

template <class V>
void runWithType(int accesses, int numThreads, int repeat, const string& name) {
  for (int N=1; N<=32; N*=2) {
    runWithTypeAndSize<V, V>(N, accesses, numThreads, repeat, name+"Default");
    runWithTypeAndSize<atomic<V>, V>(N, accesses, numThreads, repeat, name+"Atomic ");
  }
}


void runExperiment(int repeat) {
  int maxThreads = 48;
  int accesses   = 1000000;
  omp_set_num_threads(maxThreads);
  printf("OMP_NUM_THREADS=%d\n", maxThreads);
  runWithType<uint32_t>(accesses, maxThreads, repeat, "access32");
  runWithType<uint64_t>(accesses, maxThreads, repeat, "access64");
  runWithType<array<uint64_t, 2>>  (accesses, maxThreads, repeat, "access128");
  runWithType<array<uint64_t, 4>>  (accesses, maxThreads, repeat, "access256");
  runWithType<array<uint64_t, 8>>  (accesses, maxThreads, repeat, "access512");
  runWithType<array<uint64_t, 16>> (accesses, maxThreads, repeat, "access1024");
  runWithType<array<uint64_t, 32>> (accesses, maxThreads, repeat, "access2048");
  runWithType<array<uint64_t, 64>> (accesses, maxThreads, repeat, "access4096");
  runWithType<array<uint64_t, 128>>(accesses, maxThreads, repeat, "access8192");
  runWithType<array<uint64_t, 256>>(accesses, maxThreads, repeat, "access16384");
}


int main(int argc, char **argv) {
  int repeat = argc>1? stoi(argv[1]) : 1;
  runExperiment(repeat);
  printf("\n");
  return 0;
}
