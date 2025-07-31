// Project: EEL6763 Final Project
// Author: Joel Alvarez, Chase Ruskin
// File: row_wise_v2_space.c
//
// Finds the longest common subsequence (LCS) from a file storing DNA data.
//
// This file is adapted from an existing implementation (https://github.com/RayhanShikder/lcs_parallel)
// in an attempt to improve its performance on UF's HiPerGator HPC computing platform.

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"
#include "omp.h"
#include <time.h>
#include <stdint.h>
#include "lcs.h"

// Global variables
char *A_str;
char *B_str;
char *C_ustr; 
int *P_matrix;
// int *DP_Results;
int *R_prev_row;


int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int my_rank;
    int num_procs;

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if(argc <= 1) {
        printf("ERROR: No input file specified as a command-line argument.\n");
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, 104);
    }

    int len_a, len_b, len_c;
    double start_time, stop_time;

    FILE *fp = fopen(argv[1], "r");
    if(fp == NULL) {
        printf("ERROR: Failed to open file \"%s\".\n", argv[1]);
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, 104);
    }
    
    fscanf(fp, "%d %d %d", &len_a, &len_b, &len_c);

    // define the number of rows and columns in the P matrix
    int ROWS = len_c;
    int COLS = len_b+1;

    A_str = (char *)malloc((len_a+1) * sizeof(char *));
    B_str = (char *)malloc((len_b+1) * sizeof(char *));
    // the set of unique characters
    C_ustr = (char *)malloc((len_c+1) * sizeof(char *));

    fscanf(fp, "%s %s %s", A_str, B_str, C_ustr);
    
    // partition the number of units among all processes as evenly as possible
    int units_per_rank[num_procs];
    for(int i = 0; i < num_procs; i++) {
        units_per_rank[i] = get_computation_size(len_b+1, i, num_procs);
    }
    int units_per_self = units_per_rank[my_rank];

    // compute array for the displacements for each rank
    int displ_per_rank[num_procs];
    displ_per_rank[0] = 0;
    for(int i = 1; i < num_procs; i++) {
        displ_per_rank[i] = displ_per_rank[i-1] + units_per_rank[i-1];
    }

    // FIXED BUG: using malloc instead of calloc introduced errors in getting results for some cases
    // ENHANCEMENT: Saved space by not required this array at all
    // DP_Results = calloc(len_b+1, sizeof(int));
    // MORE SPACE OPT: only store NEIGHBOR_DIST neighbors of work required (including self)!
    int len_r_prev_row_size = units_per_rank[my_rank];
    for(int i = 0; i < NEIGHBOR_DIST; i++) {
        len_r_prev_row_size += units_per_rank[my_rank-i-1];
    }

    #if DEBUG > 0
        printf("P := Rank %d assigned %d chunks\n", my_rank, get_computation_size(len_c, my_rank, num_procs));
        printf("DP := Rank %d assigned %d chunks\n", my_rank, units_per_self);

        printf("displ_per_rank[%d] = %d\n", my_rank, displ_per_rank[my_rank]);
        printf("units_per_rank[%d] = %d\n", my_rank, units_per_rank[my_rank]);
        printf("Rank %d has len of R previous row: %d\n", my_rank, len_r_prev_row_size);
    #endif

    R_prev_row = calloc(len_r_prev_row_size, sizeof(int));

    P_matrix = calloc(ROWS*COLS, sizeof(int));

    begin = calloc(1, sizeof(struct timespec));
    prof_mark = calloc(1, sizeof(struct timespec));
    end = calloc(1, sizeof(struct timespec));

    // start timing immediately before distributing data
    *begin = now();

    calc_P_matrix(P_matrix, B_str, len_b, C_ustr, len_c, my_rank, num_procs);

    int result = lcs_yang(R_prev_row, P_matrix, A_str, B_str, C_ustr, len_a, len_b, len_c, my_rank, units_per_rank, displ_per_rank, num_procs, len_r_prev_row_size);

    // halt the timing
    *end = now();

    // print diagnostics about application run
    if(my_rank == CAPTAIN) {
        // print stats about resource utilization
        #pragma omp parallel
        {   
            #pragma omp single
            {
                printf("Loading DNA file \"%s\" on each of %d processes (%d threads per rank)...\n", argv[1], num_procs, omp_get_num_threads());
                #if USE_VERSION == 1
                    printf("Branching: enabled (version 1)\n");
                #else
                    printf("Branching: disabled (version 2)\n");
                #endif
            }
        }

        printf("Length of string B: %zu \n", strlen(B_str));
        printf("Length of string C: %zu\n", strlen(C_ustr));
        printf("String C is: %s\n", C_ustr);

        printf("LCS: %d\n", result);

        double exec_time = tdiff(*begin, *end);
        printf("Execution time: %lf\n", exec_time);
    }

    #if DEBUG > 2
        if(my_rank == CAPTAIN) {
            printf("P MATRIX\n");
            for(int i = 0; i < len_c*(len_b+1); i++) {
                printf("%d\t", P_matrix[i]);
            }
            printf("END P MATRIX\n");
        }
    #endif

    // deallocate pointers
    free(A_str);
    free(B_str);
    free(C_ustr);
    
    free(R_prev_row);
    free(P_matrix);
    // free(DP_Results);

    free(begin);
    free(prof_mark);
    free(end);

    MPI_Finalize();
    return 0;
}
