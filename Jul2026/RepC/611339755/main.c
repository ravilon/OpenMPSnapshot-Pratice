#include <stdio.h>
#include <stdlib.h>
// #define DEBUG
#include "common.h"

// include libraries needed for hybrid modelling of the sobel filter
#include <omp.h>
#include "mpi.h"

// defines constants used in MPI
#define MASTER 0               // taskid of first task
#define FROM_MASTER 1          // sets a message type
#define FROM_WORKER 2          // sets a message type

// NOTE: moved to below worker tasks instead of doing a function call
/*
* Apply the sobel edge detection filter algorithm on matrix input.
* Side effect: Modifies matrix output.

void sobel_edge(const unsigned M, const unsigned N, int input[M][N], int result[M][N]) {

    // Iterate from (1, 1) to (M-1, N-1)
    for(unsigned r = 1; r < M - 1; r++) {
        for(unsigned c = 1; c < N - 1; c++) {
            // Defitions of neighboring tiles:
            // p1 p2 p3
            // q1 q2 q3
            // r1 r2 r3

            const int p1 = input[r-1][c-1], // Top left
            p2 = input[r-1][c], // Top center
            p3 = input[r-1][c+1], // Top right
            r1 = input[r+1][c-1], // Bottom left
            r2 = input[r+1][c], // Bottom center
            r3 = input[r+1][c+1], // Bottom right
            q1 = input[r][c-1], // Center left
            // q2 = center, frame of reference
            q3 = input[r][c+1]; // Center right

            const int horizontal = abs((p1 - r1) + 2 * (p2 - r2) + (p3 - r3));
            const int vertical = abs((p1 - p3) + 2 * (q1 - q3) + (r1 - r3));

            result[r][c] = horizontal + vertical;
        }
    }
}
*/


int main(int argc, char *argv[]) {
    DP("Starting up...\n");

    // inits variables
    int taskid, numtasks, numworkers, dest, mtype;  // MPI variables
    MPI_Status status;
    int rows, cols, averow, extra, offset, source;  // matrix variables
    cols = 5000;                                    // col size of input mat

    // inits MPI data and runs task error check
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
    MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
    numworkers = numtasks - 1;

    // initializes I/O matricies
    const unsigned img_M = 5000;
    const unsigned img_N = 5000;
    int input[img_M][img_N];
    int output[img_M][img_N];

    /************************** Master Rank **************************/
    // sends matrix data to other ranks in the set
    if (taskid == MASTER) {

        // const unsigned img_M = 5000;         // removed from p1
        // const unsigned img_N = 5000;         // removed from p1
        // int input[img_M][img_N];             // removed from p1

        // loads input image matrix
        load_image("../input.txt", img_M, img_N, input);
        DBG(print_matrix(input, img_M, img_N);)
        
        // prints message confirming master rank had loaded in input.txt
        printf("rank %d has loaded in the image matrix\n", taskid);

        // inits parameters to send matix data to worker tasks
        averow = img_M/numworkers;
        extra = img_M%numworkers;
        offset = 0;

        // starts the timer
        struct timespec start = now();

        // sends matrix data to workers tasks
        mtype = FROM_MASTER;
        for (dest=1; dest<=numworkers; dest++) {
            rows = (dest <= extra) ? averow+1 : averow;   	
            printf("Sending %d rows to task %d offset=%d\n",rows,dest,offset);
            MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&input[offset][0], rows*cols, MPI_INT, dest, mtype, MPI_COMM_WORLD);      
            offset = offset + rows;
        }

        // recieves matrix data from workers tasks
        mtype = FROM_WORKER;
        for (int i=1; i<=numworkers; i++) {
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&output[offset][0], rows*cols, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);      
            printf("Received results from task %d\n",source);
        }

        // stops the timer
        struct timespec end = now();

        // prints output matrix and calculates elapsed runtime
        DBG(print_matrix(output, img_M, img_N);)
        save_image("../output.txt", img_M, img_N, output);
        double elapsed_time = tdiff(start, end);
        printf("Elapsed time: %.8f sec\n", elapsed_time);
    }


    /************************** Worker Ranks **************************/
    // rcvs matrix chunks and runs sobel algorithim, 
    // sends results from the alg back to master
    // each matrix chunk should be 5000/3 by 5000/3 for the 3 worker tasks
    if (taskid > MASTER) {

        // receives the matrix chunks and row/offset data from the master
        mtype = FROM_MASTER;
        MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&input, rows*cols, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);

        // prints the offset and row count for a task
        printf("task %d has %d rows and an offset of %d\n",taskid,rows,offset);

        // defines a parallel thread workspace
        #pragma omp parallel
        {
            
            // stores thread id# in a variable
            int tid = omp_get_thread_num();
            
            // runs sobel algorithim using openMP's thread modelling
            #pragma omp for schedule(static)
            for(unsigned r = 1; r < rows - 1; r++) {
                for(unsigned c = 1; c < cols - 1; c++) {
                    // Defitions of neighboring tiles:
                    // p1 p2 p3
                    // q1 q2 q3
                    // r1 r2 r3

                    const int p1 = input[r-1][c-1], // Top left
                    p2 = input[r-1][c], // Top center
                    p3 = input[r-1][c+1], // Top right
                    r1 = input[r+1][c-1], // Bottom left
                    r2 = input[r+1][c], // Bottom center
                    r3 = input[r+1][c+1], // Bottom right
                    q1 = input[r][c-1], // Center left
                    // q2 = center, frame of reference
                    q3 = input[r][c+1]; // Center right

                    const int horizontal = abs((p1 - r1) + 2 * (p2 - r2) + (p3 - r3));
                    const int vertical = abs((p1 - p3) + 2 * (q1 - q3) + (r1 - r3));

                    output[r][c] = horizontal + vertical;
                }
            }
        }

        // sends results back to master
        mtype = FROM_WORKER;
        MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&output, rows*cols, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    }


    // calls final return
    MPI_Finalize();
    printf("Goodbye.\n");
    return 0;
}

