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
{ /*Still, the output might get jumbled up, because cout is a shared resource. critical only ensures that this block is only executed by one thread at a time. 
The if block might still happen at the same time.*/
std::cout << "Hello World from thread = " << tid << std::endl;
}
/* Only the master thread does this */
if (tid == 0) 
{
nthreads = omp_get_num_threads();
std::cout << "Number of threads = " << nthreads << std::endl;
}

}  /* All threads join master thread and terminate */
}
