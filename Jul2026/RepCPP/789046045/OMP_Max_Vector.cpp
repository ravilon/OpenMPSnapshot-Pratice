#include <omp.h>
#include <ctype.h>
#include <stdio.h>
#include <iostream>
using namespace std;
// defining a function to find the maximum of the vector
void find_max(int n, double* vector, int numThreads)
{
  double max_element = vector[0];     // intializing the maximum value of the generated vector
  double itime, ftime, exec_time;     // Declaring some variables to calculate the time of execution
  omp_set_dynamic(0);                 // Explicitly disable dynamic teams
  omp_set_num_threads(numThreads);    // Use amount of (numThreads) threads for all consecutive parallel regions
  itime = omp_get_wtime();            // Starting calculating the time 
  #pragma omp parallel 
  {
    #pragma for reduction(max : max_element) //Use Parallel For Reduction to find the mas among the vector
    for (int idx = 0; idx < n; idx++)
        max_element = max_element > vector[idx] ? max_element : vector[idx];
  }
  ftime = omp_get_wtime();            // Ending calculating the time
  exec_time = ftime - itime;          // Calculating how much time spent
  printf("Time taken %f\t", exec_time);   // Printing the execution time
  printf(" Max value %f ", max_element);  //Printing the Max value found
}
int main(int argc, char* argv[]) {
  // Reading the first arg : Length of vector
  int n = atoi(argv[1]);
  // Generating the vector
  double* vector = new double[n];
  for (long int i = 0; i <= n; i++) {
    vector[i] = (double)rand() / RAND_MAX;
    // Printing the generated vector, just for checking the 
    //correctness of the algorithm in the first run
	  //printf("vector[%d] = %f\n", i, vector[i]);
  }  
  // Calling the find_max function ten different times with num_threads changing from 1 to 10
  for (int j=1; j<= 10; j++)
  {
    printf("\nNum of threads %d \t", j); 
    find_max(n,vector,j);
  }
  delete[] vector;  
}