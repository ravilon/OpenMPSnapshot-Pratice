// include necessary libraries
#include <iostream>
#include <stdlib.h>
#include <cstdlib>
#include <cmath>
#include <omp.h> // OpenMP library

// define namespace
using namespace std;

// prototype function(s)
double range_rand_double(double low, double high);

// start main
int main(int argc, char* argv[])
{
	double start_time = omp_get_wtime(); // wall time at the start of the program


	if(argc < 3)
	{
		cerr << "Please enter a command of the form: ./<prog_name> <matrix_size> <num_of_threads>" << endl; //print error
		return -1; // return error
	}


	// defining necessary variables
	int mat_size = atoi(argv[1]); // user specified matrix size
	int thread_count = atoi(argv[2]); // user specified thread count
	int chunk_size = mat_size / thread_count;
	double double_lower_bound = 0;
	double double_upper_bound = 500000;


	srand(time(NULL)); // randomize seed based on current time


	cout << endl << "Creating square matricies of size " << mat_size << "..." << endl;
	double A[mat_size][mat_size];
	double B[mat_size][mat_size];
	double C[mat_size][mat_size];


	cout << endl << "Initializing random values..." << endl;
	for(int i = 0; i < mat_size; i++)
	{
		for(int j = 0; j < mat_size; j++)
		{
			A[i][j] = range_rand_double(double_lower_bound, double_upper_bound);
			B[i][j] = range_rand_double(double_lower_bound, double_upper_bound);
		}
	}


	double make_time = omp_get_wtime() - start_time; // seqential creation and initialization time


	cout << endl << "Generating matrix C by square matrix multiplication (C = A•B) with " 
	<< thread_count << " threads and a chunk size of " << chunk_size << "..." << endl;
	#pragma omp parallel num_threads(thread_count)
	{
		//private variables to each thread
		double curr_cell;
		int start, end;
		int thread_ID, total_threads;
		thread_ID = omp_get_thread_num(); // unique thead
		total_threads = omp_get_num_threads(); // quantity of total threads

		start = (chunk_size * thread_ID);
	  end = (start + chunk_size - 1);
	  if(thread_ID == (total_threads - 1))
	  {
	    end = mat_size - 1;
	  }
		for(int i = start; i <= end; i++)
		{
	    for(int j = 0; j < mat_size; j++)
	    {
	      curr_cell = 0; // reset curr_cell value 
	      for(int k = 0; k < mat_size; k++)
	      {
	        curr_cell += (A[i][k]*B[k][j]);
				}
	      C[i][j] = curr_cell; 
	      // cout << C[i][j] << "\t"; // for printing each element
			}
	    // cout << endl; // for ending each row
		}
	}


	double mult_time = omp_get_wtime() - make_time - start_time; // matrix mutiplication time


	double seq_time = make_time;
	double parallel_time = mult_time;


	cout << endl << "Matrices of square size " << mat_size << " were sequentially created & initialized in "
			 << seq_time << " seconds." << endl;

	cout << endl << "Matrices of square size " << mat_size << " were multiplied in parallel in " 
			 << parallel_time << " seconds." << endl;


	return 0;
}


double range_rand_double(double low, double high)
{
  double range = high - low; // get the range of values
  return (rand() / double (RAND_MAX) * (range - 1)) + low; // return a random double between high and low
}
