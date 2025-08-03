/* --------------------------------------------------------------------
* Shaun Jorstad ~ Comp 233 ~ Capstone Project
* 
* Runs jacobi iterations on a an array of floats spawning 4 processes 
* distributed evenly across 4 nodes, each spawning 4 threads.
* 
* Code forked from Argonne National Laboratory
* available at: https://www.mcs.anl.gov/research/projects/mpi/tutorial/mpiexmpl/src/jacobi/C/main.html 
*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <time.h>

#define MAX_ITERATIONS 500000     // max iterations for convergence
#define EPSILON .01               // required difference value
#define NORTH 100                 // constant temp of North edge
#define SOUTH 100                 // constant temp of South edge
#define EAST 0                    // constant temp of East edge
#define WEST 0                    // constant temp of West edge
#define WIDTH 1000                // width of the plate
#define HEIGHT 1000               // height of the plate
#define MASTER 0                  // rank of the master node

void runIterations();           

void printImage(float* plate);

void printHeader(int iterCount);

/* --------------------------------------------------------------------
* runs the simulation that generates the heat map
*
*/
int main(int argc, char** argv) {
MPI_Init(&argc, &argv);             // initializes mpi
int worldRank, worldSize;           // mpi variables
double start, stop;                 // timers
int numProcs, numThreads;           // number of proc and threads
int iterCount;                      // number of iterations to conv.   

MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

// Simulation -----------------------------------------------------
if (worldRank == MASTER) {
printf("Shaun Jorstad ~ Thermal Plate conduction\n");
start = MPI_Wtime();            // starts the timer
}

runIterations();

if (worldRank == MASTER) {
stop = MPI_Wtime();             // stops the timer 
printf("Simulation took: %f seconds\n", (stop-start));
printf("-----Normal Termination-----\n");
}

MPI_Finalize();
return 0;
}

/* --------------------------------------------------------------------
* Function:    runIterations
* 
* Runs the Jacboi iterations on 4 processes, each with 4 threads. 
* 
* Each process creates an appropriately sized array of floats, sends 
* and receives the neccessary rows from neighboring processes, and 
* calculates the next iteration of the plate. This loops until the 
* difference between the next iteration and current iteration of 
* the plate is within .01
* 
* Each process sends their final chunk of the plate to master. 
* Master than converts the temperatures to colors and prints to the 
* file. Ideally, each process would convert their values to temperatures 
* and would send an array of integers. But this process doesn't take long 
* and wouldn't speed the simulation up by much. 
*/
void runIterations() {
int     row, col;               // array iterators
int     iterCount;              // iteration count to conv.
int     rowStart, rowEnd;       // current process start and stop
int     averageColor;           // average temp of border temp.
int     worldRank, worldSize;   // mpi variables
int     procIndex;              // iterates over processes
int     processHeight;          // number of rows each process calculates
double  diffNorm, gDiffNorm;    // global and local difference between arrays
MPI_Status status;              // mpi variable

MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

averageColor = (NORTH + SOUTH + EAST + WEST) / 4;
iterCount = 0;

// calculates process row start and stop index --------------------
// takes into account running 1000x1000 on 12 and 16 nodes
processHeight = (HEIGHT/ worldSize);
rowStart = 1;
rowEnd = rowStart + processHeight;
if (worldRank == MASTER) {
rowEnd -= 1;
}
else if (worldRank == worldSize-1) {
rowEnd -= 1;
}

// allocates and initializes the 2d array -------------------------
float *current = (float*) malloc(WIDTH * (processHeight + 2) * sizeof(float)); 
float *next = (float*) malloc(WIDTH * (processHeight +2) * sizeof(float));

// initiate the proper values in the 2d array ---------------------
// each process fills array with the average color
for (row = rowStart; row < rowEnd; row++) {
for (col = 0; col < WIDTH; col++) {
*(current + (row * WIDTH) + col) = averageColor;
*(next + (row * WIDTH) + col) = averageColor;
}
}
//master and the last slave write their red row
if (worldRank == MASTER) {
for (col = 0; col < WIDTH; col++) {
*(current + col) = 100;
*(next + col) = 100;
}
}
if (worldRank == worldSize -1) {
for (col = 0; col < WIDTH; col++) {
*(current + (250 * WIDTH) + col) = 100;
*(next + (250 * WIDTH) + col) = 100;
}
}
// each process write's the blue border
for (row = rowStart; row < rowEnd; row++) {
*(current + (row * WIDTH)) = 0;
*(current + ((row + 1) * WIDTH) -1) = 0;
*(next + (row * WIDTH)) = 0;
*(next + ((row + 1) * WIDTH) -1) = 0;
}

do {
/* Send down unless I'm at the top, then receive from below */
if (worldRank < worldSize - 1) {
MPI_Send( current + ((rowEnd -1) * WIDTH), WIDTH, MPI_FLOAT, worldRank + 1, 0, MPI_COMM_WORLD); 
}
if (worldRank > 0) {
MPI_Recv( current, WIDTH, MPI_FLOAT, worldRank - 1, 0, MPI_COMM_WORLD, &status );
}

/* Send up unless I'm at the bottom */
if (worldRank > 0) {
MPI_Send( current + WIDTH, WIDTH, MPI_FLOAT, worldRank - 1, 1, MPI_COMM_WORLD );
}
if (worldRank < worldSize - 1) {
MPI_Recv( current + (rowEnd * WIDTH), WIDTH, MPI_FLOAT, worldRank + 1, 1, MPI_COMM_WORLD, &status );
}

/* Compute new values (but not on boundary) */
iterCount ++;
diffNorm = 0.0;
// spawns a team of threds to calculate the next iteration of the plates
# pragma omp parallel for reduction(+:diffNorm) num_threads(4) private(row, col) shared(current, next)
for (row = rowStart; row < rowEnd; row++) {
for (col = 1; col < WIDTH -1; col++) {
*(next + (row * WIDTH) + col) = (*(current + (row * WIDTH) + (col -1)) + *(current + (row * WIDTH) + (col + 1)) + *(current + ((row - 1) * WIDTH) + col) + *(current + ((row+1) * WIDTH) +col)) / 4;
diffNorm += (*(next + (row * WIDTH) + col) - *(current + (row * WIDTH) + col)) * (*(next + (row * WIDTH) + col) - *(current + (row * WIDTH) + col));
}
}

// changes the current plate to be the new plate. re-uses the previously allocated plate for the next iteration
float* tempPointer = current;
current = next;
next = tempPointer;

MPI_Allreduce( &diffNorm, &gDiffNorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
gDiffNorm = sqrt( gDiffNorm );
if (worldRank == 0 && iterCount % 1000 == 0) {
printf( "At iteration %d, diff is %e\n", iterCount, gDiffNorm );
}
} while (gDiffNorm > EPSILON && iterCount < MAX_ITERATIONS);

// master receives each part of the image and prints to the file
if (worldRank == MASTER) {
printHeader(iterCount);
printImage(current, iterCount);
for (procIndex = 1; procIndex < 4; procIndex++) {
// receive image
MPI_Recv(current, WIDTH*(processHeight), MPI_FLOAT, procIndex, 1, MPI_COMM_WORLD, &status);
// print image
printImage(current, 0);
}
}
else {
MPI_Send((current + (WIDTH)), WIDTH*(processHeight), MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
}

// Garbage Cleanup ------------------------
free(current);
free(next);
}

/* --------------------------------------------------------------------
* prints the header of the ppm file
*
*      type        name            description
*      ----        ----            ----------- 
*      int         iterCount       The number of iterations the plate 
*                                  underwent before convergence
*/
void printHeader(int iterCount) {
char* path = "./heatmap.ppm";   // path to the image
FILE *out;                      // file pointer
out = fopen(path, "w+");        // opens the stream
int r = 0;                      // iterators
int c = 0;
int pixel;
float red, blue;                // color values 0..255

// prints header to the ppm file
fprintf(out, "P3\n");
fprintf(out, "# Shaun Jorstad ~ Generated Thermal Plate\n");
fprintf(out, "# Forked source code from Argonne National Laboratory\n");
fprintf(out, "# Executed in: %d iterations\n", iterCount);
fprintf(out, "%d %d 255\n", WIDTH, HEIGHT);
fclose(out);    // closes stream
}

/* --------------------------------------------------------------------
* prints the image to a ppm file
*
*      type        name            description
*      ----        ----            -----------
*      float*     plate           The 2d array of floats representing 
*                                  the plate being printed to an image
*/
void printImage(float* plate) {
char* path = "./heatmap.ppm";   // path to the image
FILE *out;                      // file pointer
out = fopen(path, "a+");        // opens the stream
int r = 0;                      // iterators
int c = 0;
int pixel;
float red, blue;                // color values 0..255

// prints pixels to image file
while (r < 250) {
while (c < WIDTH) {
for (pixel = 0; pixel < 5; pixel++) {
red = (*(plate + (r*WIDTH) + c)/ 100.0) * 255.0;
blue = 255 - red;
fprintf(out, "%.0f 0 %.0f\t", red, blue);
c += 1;
}
fprintf(out, "\n");
}
r++;
c = 0;
}
fclose(out);    // closes stream
}
