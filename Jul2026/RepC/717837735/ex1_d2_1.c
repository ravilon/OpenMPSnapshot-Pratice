#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

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
		
		// Part D (Lock) ============================================================
		int *S = malloc(sizeof(int)*P*2); // 2 for each thread
		int tid;
				
		timer[6] = omp_get_wtime();

		// Execute in parallel....
		#pragma omp parallel shared(N, B, S, P) private(i, o, tid)
		{
			tid = omp_get_thread_num();
			
			// Each thread will find its local minimum.
			// Initialize S with the largest positive 4-byte integer
			S[tid] = 0x7FFFFFFF;
			S[tid + P] = 0x7FFFFFFF;
			
			// Split lines (N) to threads
			#pragma omp for schedule(static, 1) // S[tid]= threds's minimum; S[tid+P]= Current line's min.
			for(i = 0; i < N; i++){ // For each line a thread gets...
				for(o = 0; o < N; o++){ // Find minimum in line
					S[tid+P] = (S[tid+P] > B[i*N+o]) ? B[i*N+o] : S[tid+P]; 
				}
				S[tid] = (S[tid] > S[tid+P]) ? S[tid+P] : S[tid]; // Compare minimum of last line with local minimum
			}
			
			#pragma omp barrier
			
			// Every thread checks S[P] and writes to it
			#pragma omp critical
			{
				 S[P] = (S[tid] < S[P]) ? S[tid] : S[P];	
			}	
		
		}
		timer[7] = omp_get_wtime();
		
		printf(" -> Min in B: %d\n", S[P]);
		printf("Time D: %6f\n\n", timer[7]-timer[6]);
				
		
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
		
		free(B);
		free(S);
		printf("\n");
		
	}
	else{
		printf("usage: ex1_2 [Test Number]\n\n");
	}
	return 0;
}

