#include <iostream>
#include <omp.h>
using namespace std;

int main() {
  int a[10000]; // shared among all threads since declared and defined in master
                // thread worflow.
  for (int i = 0; i < 10000; i++)
    a[i] = (i + 2 * rand() + 10000) % 10000;
  int b; // shared among all threads since declared and defined in master thread
         // worflow.
  // default num of threads for this program if no num_threads() specified for
  // any parallel reigon, num_threads() can be specified for a parallel reigon
  // and different parallel reigons can have different numbers of threads
  // working on it.
  int c;
#pragma omp parallel num_threads(4) shared(a)                                  \
    shared(b) private(c) // this reigon will use only 4 threads, a is shared and
                         // b becomes private for each thread.
  {
    int id = omp_get_thread_num(); // unique for each thread since defined in
                                   // parallel reigon
    b = a[id]; // here all threads are trying to write to b, we can not be sure
               // of what value b will take. Also called as concurrent write to
               // b. Data race condition. This should be avoided, unless done
               // intentionally.
    c = a[id]; // unlike b, c is private to each thread, meaning each thread has
               // its own copy of c, and no other thread is allowed to access
               // it.
    cout << "b is " << b << " for thread : " << id << "\n";
  }
}