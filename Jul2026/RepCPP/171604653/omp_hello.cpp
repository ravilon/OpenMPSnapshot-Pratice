// include necessary libraries
#include <iostream>
#include <stdlib.h>
#include <omp.h> // OpenMP library

using namespace std;

int main(int argc, char* argv[])
{
	double start_time = omp_get_wtime(); // wall time at the start of the program

	if(argc < 2)
	{
		cerr << "Please enter a command of the form: ./<prog_name> <num_of_threads>" << endl; //print error
		return -1; // return error
	}

	int thread_count = atoi(argv[1]); // user specified thread count

	#pragma omp parallel num_threads(thread_count) // run block with a total of <thread_count> threads
	{
		int thread_ID, total_threads; // private variables to each thread
		thread_ID = omp_get_thread_num(); // which thread?
		total_threads = omp_get_num_threads(); // quantity of total threads

		#pragma omp critical // critical section only runs one thread at a time (serial/sequential)
		{
			if (thread_ID == 0) // if master thread
			{
				cout << "Threads are 0-indexed with the master thread as thread 0." << endl;
				cout << "Total number of threads: " << total_threads << endl;
			}

			cout << "Hello from " << thread_ID << " of " << total_threads << "." << endl;
		}
	}

	double run_time = omp_get_wtime() - start_time; // total run time
	cout << "Execution time: " << run_time << "seconds." << endl;
	return 0;
}