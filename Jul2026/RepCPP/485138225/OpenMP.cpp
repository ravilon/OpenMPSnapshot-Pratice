#include <iostream>
#include <omp.h>

using namespace std;

int main(int argc, char** argv){
 
 #pragma omp parallel num_threads(4)
 // create 4 threads and region inside it will be executed by all threads.
 {
   #pragma omp critical
   // allow one thread at a time to access below statement
   cout<<"Threads ID is OpemMP stage 1 = "<<omp_get_thread_num()<<endl; 
 } // here all thread get merged into one thread id
 
 cout<<"I am Muhammad Allah Rakha"<<endl;
 
 #pragma omp parralel num_threads(2)
 // create 2 threads
 {
   cout<<"Thread ID in OpenMP stage 2 = "<<omp_get_thread_num()<<endl;
 }

}
