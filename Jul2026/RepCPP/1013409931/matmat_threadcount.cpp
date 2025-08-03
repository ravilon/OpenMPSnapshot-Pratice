#include <random>
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

int main(int argc, char **argv) {
  int N;
  if (argc > 1)
    N = atoi(argv[1]);
  else {
    N = 2048;
  }
  //Create 3 matrices
  int num_of_threads = 4;
  std::vector<float> a(N*N);
  std::vector<float> b(N*N);
  std::vector<float> c(N*N);

  //Create random generator with 0 seed
  std::default_random_engine generator(0);
  //Create distribution - float values between 0.0 and 1.0
  std::uniform_real_distribution<float> distribution(0.0,1.0);
  //Fill matrices with random values
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      a[i*N+j] = distribution(generator);
      b[i*N+j] = distribution(generator);
    }
  }
  omp_set_num_threads(num_of_threads);
  std::vector<std::vector<int>> rows_foreach_thread(num_of_threads);
  auto t1 = std::chrono::high_resolution_clock::now();
  //implement matrix-matrix multiplication here
  #pragma omp parallel for
  for (int i = 0; i<N; i++){
    int thread_id = omp_get_thread_num();
    #pragma omp critical
    {
      rows_foreach_thread[thread_id].push_back(i);
    }
    for (int j = 0; j<N; j++)
      for (int k = 0; k<N; k++)
        c[i*N+j] += a[i*N+k] * b[k*N+j];
  }
  auto t2 = std::chrono::high_resolution_clock::now();

  //checksum
  float sum = 0;
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      sum += c[i*N+j];
  std::cout << sum << std::endl;
  std::cout << "took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
              << " milliseconds\n";
  std::cout << 2*N*N*N/std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()/1000.0/1000.0 << " GFLOPS/s" << std::endl;
  for(int t=0; t< num_of_threads; t++){
    int tot_rows = rows_foreach_thread[t].size();
    std::cout << "Thread " << t << " (Total rows: " << tot_rows << "): ";

    for (int i = 0; i < 3; i++){
          std::cout << rows_foreach_thread[t][i] << " ";
      }
          std::cout << "...";
      std::cout << std::endl;
  }
  
  return 0;
}
