#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <omp.h>
using namespace std;
// Defining Auxiliary functions
// Declaring function to allocate memory for n*n array
double** malloc_array(int n)
{
	double** array = new double*[n];
	for (int i = 0; i < n; i++)
		array[i] = new double[n];
	return array;
}
// Declaring a function to delete generated arrays
void delete_array(double** array, int n)
{
	for (int i = 0; i < n; i++)
		delete[] array[i];
	delete[] array;
}
// Declaring function to initialize an n*n array with random numbers in the range [0,1]
void generate_array(double** array, int n)
{
	srand(time(NULL));
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			array[i][j] = double(rand()) / RAND_MAX;
}
// Declaring function to initialize an n*n array with zeros values
void generate_array_zero(double** array, int n)
{
	srand(time(NULL));
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			array[i][j] = 0.0;
}
// Declaring function to print the arrays for checking the algprithm
void print_array(double** array, int n)
{
    for(int i = 0; i < n; i++)
    {
      for(int j = 0; j < n; j++)
        cout << array[i][j] << " ";
      cout << endl;
    }
}

int main(int argc, char** argv)
{
  //Defining loops' indecies
  int i, j, k;
  // Reading the first argument: size of array
	const int N = atoi(argv[1]);
  // Defining three arrays A,B and C to hold the result
	double** A, **B, **C;
  // Memory allocating and intializing for arrays A, B, C 
	A = malloc_array(N); generate_array(A, N);
	B = malloc_array(N); generate_array(B, N);
	C = malloc_array(N); generate_array_zero(C, N);	
	// Defining a variable to hold the time before starting the execution 
  double t_start;
  // Printing the generated arrays only for the first launch to check the correctness of the algoritm
  //cout<<"Array A"<< endl;
  //print_array(A,N);
  //cout<<"******\n Array B"<< endl;
  //print_array(B,N);
  //cout<<"**************"<<endl;
  // First Order: array multiplication with order ijk
  cout << "ijk multiplication" << endl;
  // Calculating with one stream
  t_start = omp_get_wtime();
	for ( i = 0; i < N; i++)
		for ( j = 0; j < N; j++)
			for ( k = 0; k < N; k++)
				C[i][j] += A[i][k] * B[k][j];
  //cout<<"One stream Result"<<endl;
  //print_array(C,N);
  //cout<<"**************"<<endl;
	double t_ijk_1stream = omp_get_wtime() - t_start;	
  // Calculating with different number of threads
    for(int num_threads = 2; num_threads <= 10; num_threads++)
    {
        generate_array_zero(C, N);
        t_start = omp_get_wtime();
        #pragma omp parallel for num_threads(num_threads) shared(A, B, C) private(i, j, k)
            for (i = 0; i < N; i++)
                for (j = 0; j < N; j++)
                    for (k = 0; k < N; k++)
                        C[i][j] += A[i][k] * B[k][j];
        double t_ijk_parallel = omp_get_wtime() - t_start;
        cout << "Multiplication time with " << num_threads << " threads: " << t_ijk_parallel << " seconds, efficiency: " << t_ijk_1stream / t_ijk_parallel << endl;
    }
  //cout<<"******\nParallel 10 threads Result"<<endl;
  //print_array(C,N);
	cout << "*********" <<  endl;
//***************************************************//
// Second Order: array multiplication with order jki
  cout << "jki multiplication" << endl;
	generate_array_zero(C, N);
	t_start = omp_get_wtime();
	for ( j = 0; j < N; j++)
		for ( k = 0; k < N; k++)
			for ( i = 0; i < N; i++)
				C[i][j] += A[i][k] * B[k][j];
	double t_jki_1stream = omp_get_wtime() - t_start;

    for(int num_threads = 2; num_threads <= 10; num_threads++)
    {
        generate_array_zero(C, N);
        t_start = omp_get_wtime();
        #pragma omp parallel for num_threads(num_threads) shared(A, B, C) private(i, j, k)
            for (j = 0; j < N; j++)
                for (k = 0; k < N; k++)
                    for (i = 0; i < N; i++)
                        C[i][j] += A[i][k] * B[k][j];
        double t_jki_parallel = omp_get_wtime() - t_start;
        cout << "Multiplication time with " << num_threads << " threads: " << t_jki_parallel << " seconds, efficiency: " << t_jki_1stream / t_jki_parallel << endl;
    }
  //cout<<"******\nParallel 10 threads Result"<<endl;
  //print_array(C,N);
	cout << "***********" <<  endl;

//***************************************************//
// Third Order: array multiplication with order kji
  cout << "kji multiplication" << endl;
	generate_array_zero(C, N);
	t_start = omp_get_wtime();
	for ( k = 0; k < N; k++)
		for ( j = 0; j < N; j++)
			for ( i = 0; i < N; i++)
				C[i][j] += A[i][k] * B[k][j];
	double t_kji_1stream = omp_get_wtime() - t_start;

    for(int num_threads = 2; num_threads <= 10; num_threads++)
    {
        generate_array_zero(C, N);
        t_start = omp_get_wtime();
        #pragma omp parallel for num_threads(num_threads) shared(A, B, C) private(i, j, k)
            for (i = 0; i < N; i++)
                for (k = 0; k < N; k++)
                    for (j = 0; j < N; j++)
                        C[i][j] += A[i][k] * B[k][j];
        double t_kji_parallel = omp_get_wtime() - t_start;
        cout << "Multiplication time with " << num_threads << " threads: " << t_kji_parallel << " seconds, efficiency: " << t_kji_1stream / t_kji_parallel << endl;
    }
    //cout<<"******\nParallel 10 threads Result"<<endl;
    //print_array(C,N);
    cout << "********" <<  endl;

// Freeing memory occupied by arrays A, B, C
	delete_array(A, N);
	delete_array(B, N);
	delete_array(C, N);
	return 0;
}