
// Import Library

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

// Constant

#define STEPS 50
#define FEATURES 9

// Save Location

void Save_Coordinates(int *seq) {
    seq[1] = seq[7];
    seq[2] = seq[8];
}

// Find Next Location

void Update_Coordinates(int *seq,int N) {

    seq[7]=(((seq[3]*seq[7]+ seq[4]) % N) + N) % N;
    seq[8]=(((seq[5]*seq[8]+ seq[6]) % N) + N) % N;
}

// Equations Of Motion

// a', b', c', d'

void Update_Equations_Of_Motion(int *array) {

    // a'
    if ((array[1] - array[8]) < 0 && (array[1] - array[8]) % 10 != 0)
			array[3] = (((array[1] - array[8]) % 10) + 10) % 10;
    else if ((array[1] - array[8]) > 0 && (array[1] - array[8]) % 10 != 0)
			array[3] = -1 * ((array[1] - array[8]) % 10);

    // b'
    if ((array[1] - array[2]) < 0 && (array[1] - array[2]) % 30 != 0)
			array[4] = (((array[1] - array[2]) % 30) + 30) % 30;
    else if ((array[1] - array[2]) > 0 && (array[1] - array[2]) % 30 != 0)
			array[4] = -1 * ((array[1] - array[2]) % 30);

    // c'
    if ((array[2] - array[7]) < 0 && (array[2] - array[7]) % 10 != 0)
			array[5] = (((array[2] - array[7]) % 10) + 10) % 10;
    else if ((array[2] - array[7]) > 0 && (array[2] - array[7]) % 10 != 0)
			array[5] = -1 * ((array[2] - array[7]) % 10);

    // d'
    if ((array[7] - array[8]) < 0 && (array[7] - array[8]) % 30 != 0)
			array[6] = (((array[7] - array[8]) % 30) + 30) % 30;
    else if ((array[7] - array[8]) > 0 && (array[7] - array[8]) % 30 != 0)
			array[6] = -1 * ((array[7] - array[8]) % 30);
}

// Main Function

int main(int argc, char* argv[])
{
    // Input

    if (argc != 3) {
      printf("You Need To Enter Two Argument !!!\n");
      exit(EXIT_FAILURE);
    }

    //Initialization

    int N = atoi(argv[1]);
    int PARTICLES = atoi(argv[2]);


    int** sequence = (int**)malloc(PARTICLES * sizeof(int*));
    for (int i = 0; i < PARTICLES; i++)
        sequence[i] = (int*)malloc(FEATURES * sizeof(int));


    int **reds = (int**)malloc(N * sizeof(int*));
    int **blues = (int**)malloc(N * sizeof(int*));

    for (int i = 0; i < N; i++) {
		reds[i] = (int*)malloc(N * sizeof(int));
		blues[i] = (int*)malloc(N * sizeof(int));
    }

		// Initialization

    int Blue_Collisions = 0;
    int Red_Collisions = 0;
    int Blue_Red_Collisions = 0;
    int Blue_Energy = 0;
    int Red_Energy = 0;

    // Read File

    char *inputFile = "InputSample.txt";
    FILE *file = fopen(inputFile, "r");
    char * line = NULL;
    size_t len = 0;
    ssize_t read;

    if (file == NULL)
            exit(EXIT_FAILURE);
    getline(&line, &len, file);

    for (int i = 0; i < PARTICLES; i++) {

        fscanf(file, "%d,%d,%d,%d,%d,%d,%d", &sequence[i][0],&sequence[i][1],&sequence[i][2],&sequence[i][3],&sequence[i][4],&sequence[i][5],&sequence[i][6]);
        sequence[i][7] = sequence[i][1];
        sequence[i][8] = sequence[i][2];

    }
    fclose(file);

		double start_time = omp_get_wtime();

    for (int step = 0; step < STEPS; step++){

        #pragma omp parallel for schedule(static,1000)

        for (int i = 0; i < PARTICLES; i++) {
            Save_Coordinates(&sequence[i][0]);
            Update_Coordinates(&sequence[i][0], N);
        }

        #pragma omp parallel for schedule(dynamic,1000)

        for (int i = 0; i < PARTICLES; i++) {
            if (sequence[i][0] == 1){
                #pragma omp atomic
                blues[sequence[i][7]][sequence[i][8]]++;
            }
            else {
                #pragma omp atomic
                reds[sequence[i][7]][sequence[i][8]]++;
            }
        }

        #pragma omp parallel for

        for (int i = 0; i < PARTICLES; i++) {
            if(blues[sequence[i][7]][sequence[i][8]]*reds[sequence[i][7]][sequence[i][8]] >= 1 || blues[sequence[i][7]][sequence[i][8]] > 1 || reds[sequence[i][7]][sequence[i][8]]> 1) {
                Update_Equations_Of_Motion(&sequence[i][0]);
            }
        }

      	#pragma omp parallel for collapse(2) reduction(+:Blue_Energy) reduction(+:Red_Energy) reduction(+:Blue_Collisions) reduction(+:Red_Collisions) reduction(+:Blue_Red_Collisions)
        for (int i = 0; i < N; i++)
        {
           for (int j = 0; j < N; j++)
           {
           		int total = blues[i][j] + reds[i][j];
           		if (total > 1) {
           			Blue_Red_Collisions++;
           			if(blues[i][j] > 1 )
           				Blue_Collisions++;
           			if( reds[i][j] > 1)
           				Red_Collisions++;
           		}
                if(reds[i][j] >0 && blues[i][j] >0)
                {
                    Blue_Energy = Blue_Energy + blues[i][j]*5;
                    Red_Energy  = Red_Energy + reds[i][j]*5;
                }
                reds[i][j] = 0;
                blues[i][j] = 0;
            }
        }

    }

		double end_time = omp_get_wtime();
		printf("Execution Time was %lf Sec\n", end_time - start_time);

		// Output File

    FILE *Output = fopen("output_parallel_50.txt","w");

    fprintf(Output,"%s,%s,%s,%s,%s,%s,%s\n ","color","i","j","a","b","c","d");
    for(int i=0; i < PARTICLES ;i++)
    {
     fprintf(Output,"%d,%d,%d,%d,%d,%d,%d\n", sequence[i][0],sequence[i][7],sequence[i][8],sequence[i][3],sequence[i][4],sequence[i][5],sequence[i][6]);
    }
    printf("Total Collisions: %d\n", Blue_Red_Collisions);
    printf("Total Blue Collisions: %d\n", Blue_Collisions);
    printf("Total Red Collisions: %d\n", Red_Collisions);
    printf("Total Blue Energy: %d\n", Blue_Energy);
    printf("Total Red Energy: %d\n", Red_Energy);

    fclose(Output);

    return 0;
}
