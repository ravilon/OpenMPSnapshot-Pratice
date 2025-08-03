#include <omp.h>
#include <iostream>

int main()  {
int nthreads, tid;
omp_set_num_threads(4);
/* Fork a team of threads with each thread
   having a private tid variable */
#pragma omp parallel private(tid)
  {

  /* Obtain and print thread id */
  
  tid = omp_get_thread_num();
  #pragma omp critical /*Comment 1: Without this line, the threads control the console in a random order. With this: ...critical.exe*/
  {
  /*Comment 3: If we add a block from here to the end of the the if statement with { ... }, then it prints right after thread 0, but the if statement's pragma has to be removed.
    With this: ...critical3.exe. Furthermore, SAVE the file every time before compiling.*/
  std::cout << "Hello World from thread = " << tid << std::endl;

  /* Only the master thread does this */
  if (tid == 0) 
    {
    nthreads = omp_get_num_threads();
    /*#pragma omp critical*/ /*Comment 2: Without this line, the below count reaches the console randomly. With this: ...critical2.exe*/
    /*Still, we can see, that even after the thread with tid = 0 releases the consol after "Hello world...", the other threads may print
    before it enters this if statement and prints "Number of threads..." */
    std::cout << "Number of threads = " << nthreads << std::endl;
    }
  }
  }  /* All threads join master thread and terminate */
}
