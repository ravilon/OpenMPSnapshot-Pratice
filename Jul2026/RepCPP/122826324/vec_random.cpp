#include "submit.h"
#include "omp.h"
#include "stdlib.h"
#include "time.h"
#include "ex_function.h"
//#include <iostream>

//using namespace std;

float* vec_random(int n, bool normal)
{
float* rand_arr = new float[n];

// ----------------------------------------------------------
//  1.       
// ----------------------------------------------------------

#pragma omp parallel num_threads(n)
{
// Array index to generate
int index = omp_get_thread_num();

// Initialize random seed
srand(index);

// Get pseudo-random number in [1, 100]
rand_arr[index] = float(rand() % 100);

if (normal)
{
#pragma omp barrier //   .       , 
//         
//  ,  
vec_random_norm(rand_arr, n);
}

// Submit answer
Submit_test();
}
// ----------------------------------------------------------------------------------------
// --        1     --
// ----------------------------------------------------------------------------------------
return rand_arr;
}