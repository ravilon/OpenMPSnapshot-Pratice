#include <iostream>
#include <omp.h>
using namespace std;

int main() {
  // default num of threads for this program if no num_threads() specified for
  // any parallel reigon, num_threads() can be specified for a parallel reigon
  // and different parallel reigons can have different numbers of threads
  // working on it.
  omp_set_num_threads(8);
#pragma omp parallel num_threads(4) // this reigon will use only 4 threads
  {
    int id = omp_get_thread_num();
    cout << "hi : " << id << "\n";
    cout << "hello : " << id << "\n";
  }
}

// export OMP_NUM_THREADS=4 env variable based setting, useful for config files.