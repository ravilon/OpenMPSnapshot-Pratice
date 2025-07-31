#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

// Returns the binary logarithm of _n
int lb(int _n);
// Checks whether _n is a power of 2 (1 is included)
int is_mult_of_2(int _n);
// Returns the largest power of 2 that's smaller from _p
// (Used to find how many threads will be used in a binary tree algorithm)
int to_binary_tree(int _p);

// Loads the test file specified by _test_no, returning the files's data
// in N (matrix dimension), A (matrix data) and the DDM flag.
int loadTest(unsigned int _test_no, unsigned long *N, int **A, char *flag){
	FILE *fp;
	char filename[13];
	char fid_str[4];
	if(_test_no > 999)
		return -1;

	sprintf(fid_str, "%d", _test_no);
	strcpy(filename, "test/");
	strcat(filename, fid_str);

	fp = fopen((const char*)filename, "rb");
	if(fp!=NULL){
		// Read N
		fread(N, sizeof(long), 1, fp);
		// Read flag and skip padding
		fread(flag, sizeof(char), 1, fp);
		fseek(fp, 7, SEEK_CUR);
		// Allocate memory
		*A = malloc(sizeof(int)*(*N)*(*N));
		// Read A
		fread(*A, sizeof(int), (*N)*(*N), fp);
	}
	else{
		printf("Error opening output file\n");
		return 1000;
	}
	fclose(fp);
	return 0;
}

int main(int argc, char **argv){
	char passes;
	unsigned int test_no;
	unsigned long N;
	int *A;

	// Exit if argc!=2
	if(argc == 2){
		// Open and load file
		test_no = atoi(argv[1]);
		if(loadTest(test_no, &N, &A, &passes)){
			printf("error: %d is not a valid test number\n\n", test_no);
			return -1;
		}
		printf("Project I (d2.1 Implementation - Reduction Clause)\n");
		printf("Loaded test %3d: %lux%lu (%lu numbers)\n", test_no, N, N, N*N);

		double timer[8];

		// Part A ========================================================================
		unsigned long long sum_local;
		int result = 0;
		unsigned long i, o;
		unsigned int P; // Number of threads

		timer[0] = omp_get_wtime();

		// Execute the following block in parallel
		#pragma omp parallel shared(N, A, result) private(sum_local, i, o)
		{
			#pragma omp single
			{
				// Start timer and print diagnostic information
				P = omp_get_num_threads();
				printf("Threads: %d\n\n", P);
			}

			// Split lines to threads
			#pragma omp for schedule(static, 1)
			for(i = 0; i < N; i++){ // For each line a thread gets...
				sum_local = 0;
				for(o = 0; o < N; o++){ // Find sum of line
					sum_local += abs(A[(i*N)+o]);
				}

				if(sum_local-abs(A[(i*N)+i]) >= abs(A[(i*N)+i])){ // Compare sum with diagonal number
					#pragma omp atomic
						result++;
				}
			}
		}

		timer[1] = omp_get_wtime();
		printf(" -> Initial check: OK\n");
		printf("Time A: %6f\n\n", timer[1]-timer[0]);

		if(result){
			printf("Matrix not appropriate (result: %d)\n", result);
			free(A);
			return 0;
		}


		// Part B ======================================================================
		int m = 0;
		int *D = malloc(sizeof(int)*N); // Array with diagonal numbers

		timer[2] = omp_get_wtime();

		// Execute in parallel
		#pragma omp parallel shared(N, A, D, m) private(i)
		{
			// Fill D with A[i,i]
			#pragma omp for schedule(static, 1)
			for(i = 0; i < N; i++){
				D[i] = abs(A[i*N+i]);
			}

			// Find max
			#pragma omp for schedule(static, 1) reduction(max:m)
			for(i = 0; i < N; i++){
				m = m < D[i] ? D[i] : m;
			}

		}
		timer[3] = omp_get_wtime();

		printf(" -> Max: %d\n", m);
		printf("Time B: %6f\n\n", timer[3]-timer[2]);

		free(D);

		// Part C =====================================================================
		// Allocate data for B matrix
		int *B = malloc(sizeof(int)*N*N);

		timer[4] = omp_get_wtime();

		// Execute in parallel, again
		#pragma omp parallel shared(N, A, B, m) private(i)
		{
			// Fill B matrix with the differences
			#pragma omp for schedule(static, 1)
			for(i = 0; i < N*N; i++){
				B[i] = m - abs(A[i]);
			}

			// Replace numbers of the diagonal
			#pragma omp for schedule(static, 1)
			for(i = 0; i < N; i++){
				B[i*N+i] = m;
			}
		}
		timer[5] = omp_get_wtime();

		printf(" -> Matrix B: OK (will be displayed/outputed in the end)\n");
		printf("Time C: %6f\n\n", timer[5]-timer[4]);

		free(A);
		
		// Part D (Binary Tree) ========================================================
		// Each thread calculates its local minimum using omp for scheduling to find each line's minimum
		int *S = malloc(sizeof(int)*N); // Stores each line's minimum
		int *T = malloc(sizeof(int)*P); // Stores each thread's minimum
		int tid;
		
		timer[6] = omp_get_wtime();

		// In parallel...
		#pragma omp parallel shared(B, N, S, T) private(i, o, tid)
		{
			tid = omp_get_thread_num();
			
			#pragma omp for schedule(static, 1) // Fill S and T with the maximum positive value of a 4-byte int
			for(i = 0; i < N; i++){
				S[i] = 0x7FFFFFFF; 
			}
			T[tid] = 0x7FFFFFFF;
			
			// Find minimums of each line; By using OMP's for scheduling the lines are distributed evenly among threads
			#pragma omp for schedule(static, 1)
			for(i = 0; i < N; i++){ // For each line...
				for(o = 0; o < N; o++){ // ...find minimum in line
					S[i] = B[i*N+o] < S[i] ? B[i*N+o] : S[i]; // S[i] = min(B[i], current_line_min)
				}
				T[tid] = S[i] < T[tid] ? S[i] : T[tid]; // Update thread's minimum
			}
		}
		
		
		// Determine wether the number of threads is apropriate for the binary tree algorithm
		int P_bin = P; // Threads to use in binary tree
		if(!is_mult_of_2(P)){ // If there are more threads than needed...
			// Determine how many threads should actually be used
			P_bin = to_binary_tree(P);
			
			// Shrink T so that there is one number per thread in it
			#pragma omp parallel shared(P, P_bin, T) private(i, tid)
			{
				#pragma omp for schedule(static, 1)
				for(i = 0; i < P-P_bin; i++){ // Compare each for-binary thread's minimum with a remainer thread minimum (P-P_bin remainer threads)
					T[i] = T[i] < T[i+P_bin] ? T[i] : T[i+P_bin]; 
				}
			}
			printf(" -> Using %d threads for the binary tree\n", P_bin);
		}
		
				
		// Prepare for binary-tree
		int step; // Turns to 0 when the tree algorithm is completed
		int filter; // Determines which threads should continue working on the algorithm
		int stop; // Is set to 1 when a thread needs to stop performing code; If tid >= P_bin stop is set to 1 when starting
		
		#pragma omp parallel shared(N, B, S, T, P_bin) private(i, o, tid, step, filter, stop) // step and filter are private to each thread to avoid EREW incompatibilites
		{
			tid = omp_get_thread_num();

			// Star binary tree algorithm calculations
			step = lb(P_bin);
			filter = P_bin/2;
			i = P_bin/2; // note: i could be replaced with filter but for readability we'll keep them seperate
			stop = (tid >= P_bin) || (tid >= filter);
			
			// Wait until all threads are ready
			#pragma omp barrier
			
			while(step){ // All threads execute this algorithm for a predetermined number of steps
				if(stop == 0){ // Execute meaningful code...
					T[tid] = (T[tid] > T[tid+i]) ? T[tid+i] : T[tid]; // S[tid] = min(S[tid], S[tid+i])
					
				}
				
				step--;
				filter /= 2;
				i /= 2;
				stop = (tid >= P_bin) || (tid >= filter);
				
				// Wait until all threads are ready
				#pragma omp barrier
			}
		
		}
		
		timer[7] = omp_get_wtime();
		printf(" -> Min in B: %d\n", T[0]);
		printf("Time D: %6f\n\n", timer[7]-timer[6]);
		
		free(S);
		free(T);
		
		// End =======================================================================
		printf("Total Time: %6f sec\n\n", timer[7] - timer[0]);
			
		// Display B Matrix (written in file if too big or skipped entirely)
		if(N > 1024){
			printf("B Matrix is too big; Skipping dumping\n");
		}
		else{
			FILE *f = (N > 10) ? fopen("B_matrix.txt", "w") : stdout;
			for(i = 0; i < N; i++){
				for(o = 0; o < N; o++){
					fprintf(f, "%6d ", B[i*N+o]);
				}
				fprintf(f, "\n");
			}
			if(f != stdout){
				printf("B Matrix written in B_matrix.txt\n");
				fclose(f);
			}
		}
		
		free(D);
		free(B);
		printf("\n");
		
	}
	else{
		printf("usage: ex1_3 [Test Number]\n\n");
	}
	return 0;
}

// Returns the binary logarithm of _n
int lb(int _n){
	if(_n <= 0)
		return -1;

	int r = 0;
	int n = _n;
	while(n){
		r++;
		n >>= 1;
	}
	return r-1;
}

// Checks whether _n is a power of 2 (1 is included)
int is_mult_of_2(int _n){
	int i = 1;
	do{
		if(i == _n) return 1; // Multi. i with 2 got us to _n => _n is a mult. of 2
		i *= 2;
	}while(i <= _n);

	return 0;
}

// Returns the largest power of 2 that's smaller from _p
// (Used to find how many threads will be used in a binary tree algorithm)
int to_binary_tree(int _p){
	int p = _p;
	while(!is_mult_of_2(p)){
		p--;
	}

	return p;
}
