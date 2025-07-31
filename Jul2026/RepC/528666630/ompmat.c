/* Created by: Ryan Morse
//  Last Edited: 11/27/2022 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


// Use 'stringizing operator' to get variable name and return it as a string
#define getName(var) #var

// Global values for the row and column size
#define ROWS 19
#define COLS 19
#define BYTES 4096
#define MAGIC_NUMBER 0
#define TOP_LEFT 12
#define MAX_THREADS 19

char* FILE_NAME;
char* BATTER_FILE;
char* PITCHER_FILE;
int STEADY = 1;

double **M1, **M2, **M3, NEW[ROWS][COLS], BAT[ROWS][COLS], PIT[ROWS][COLS], BP[ROWS][COLS];


//  Initialize two 2D arrays using random numbers from 0.0 to 1.0
void initialize(int rows, int cols, double m1[][cols], double m2[][cols]){
		int i, j;
		double x = ((double)rand() / RAND_MAX);
		double y = ((double)rand() / RAND_MAX);
		for (i=0; i<rows; i++){
				for(j=0; j<cols; j++){
						x = ((double)rand() / RAND_MAX);
						y = ((double)rand() / RAND_MAX);
						m1[i][j] = x;
						m2[i][j] = y;
				}
		}
}

// Function to allocate 2D arrays
double** initMat()
{
        int i, j, row=19, col=19;
        double** mat = (double**) malloc(row*sizeof(double*));

        for (i=0 ; i<row ; i++)
        {       mat[i] = (double*) malloc(col*sizeof(double));  }
        return mat;
}

void cpyArr(double** mat1, double** mat2, int row, int col)
{
        int i, j;

        for (i=0 ; i<row ; i++)
        {
                for (j=0 ; j<col ; j++)
                {       mat2[i][j] = mat1[i][j];        }
        }
}

//  Display a 2D array and its variable name using double pointer
void display_dblptr(char s[], int rows, int cols, double** m){
		int i, j;

		printf("MATRIX: %s\n", s);

		for(i=0; i<rows; i++){
				for(j=0; j<cols; j++){
						printf("%2.4lf\t", m[i][j]);
				}
				printf("\n");
		}
		printf("\n");
}

// Read a .csv file into a 2D array
void csv_to_matrix(double** dm){

		int i=0, j=0;
		char file_row[4096];
		FILE* file_ptr;
		char* token;
		file_ptr = fopen(FILE_NAME, "r");
		if(file_ptr == NULL){
				perror("Error opening file.\n");
				exit (1);
		}
		else{
				printf("File opened successfully.\n");
		}

		while(fgets(file_row, 4096, file_ptr)){
				token = strtok(file_row, ",");
				while(token!= NULL){
						//printf("Value read: %s\n", token);
						dm[i][j] = atof(token);
						//printf("DM[%d][%d]= %lf\n", i, j, dm[i][j]);
						token = strtok(NULL, ",");
						++j;
				}
				free(token);
				++i;
				j=0;
				//printf("\n");
		}
		free(file_ptr);
}

// Preforms multiplication for an individual cell of the matrix
double multiply(int row, int col, double** m1, double** m2)
{
	int i;
	double total = 0;
	
	for(i=0; i<ROWS; i++)
	{	total += m1[row][i]*m2[i][col];	}
	return total;
}

int main(int argc, char *argv[])
{
	char hold[1024];
	double timeTot, startT, stopT, accm;
	double** swp;
	int Thrds = atoi(argv[1]), i, j, k;
	
	if (Thrds > MAX_THREADS || !Thrds)
	{	Thrds = MAX_THREADS;	}

	omp_set_num_threads(Thrds);
	FILE_NAME = "mat.csv";		
	M1 = initMat();
	M2 = initMat();
	M3 = initMat();
	
	csv_to_matrix(M1);
	cpyArr(M1, M2, ROWS, COLS);
	//display_dblptr(hold, ROWS, COLS, M1);
	//display_dblptr(hold, ROWS, COLS, M2);
	startT = omp_get_wtime();
	
	do
	{
	STEADY=1;
	#pragma omp parallel for private(i,j,k,accm) shared(M1,M2,M3,STEADY)
	for (i=0 ; i<ROWS ; i++)
	{
		for (j=0 ; j<COLS ; j++)
		{
			accm = 0;
			for (k=0 ; k<ROWS ; k++)
			{	accm += M1[i][k] * M2[k][j];	}
			
			M3[i][j] = accm;		
			if (j<12 && STEADY>0 && accm>0)
			{	
				#pragma omp atomic write
				STEADY=0;
			}
		}
	}
					
	if (!STEADY)
	{
		swp = M3;
		M3 = M1;
		M1 = swp;
	}
	} while(!STEADY);
	
	stopT = omp_get_wtime();
        timeTot = stopT-startT;
	
	display_dblptr(hold, ROWS, COLS, M3);
	printf("Time Total: %f\n",timeTot);	
	free(M1);
	free(M2);
	free(M3);
	return 0;
}




