#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "header.h"

// parameters for heat transfer
static struct Parms
{ 
float cx;
float cy;
} parms = {0.1, 0.1};

int main(int argc, char **argv)
{
char processor_name[MPI_MAX_PROCESSOR_NAME];
double start_time, end_time;										// Master process time counters
int myrank;															// Rank of process
int numprocs;														// Number of processes
int N;																// Grid blocks per process
int BLOCKS;															// Total grid blocks
int STEPS;															// Time steps
int CHECK;															// Period of stability check
int DIM;															// Block cells
int provided;														// Provided threads by MPI_Init_thread
int mpi_dim[2];														// MPI grid topology dimension
int period[2];														// MPI grid topology cyclic indicator
int reorder;														// MPI grid reorder indicator
int left, right, top, bottom;										// Appropriate neighbors
int cur_grid;														// IF 0 -> fgrid is current temperature grid ELSE lgrid ...
int counter;														// Step counter
int namelen;														// Length of host name
int i, j, k, l;														// Thread - private counters
float *fgrid, *lgrid, *temp_grid, *next_grid;						// Two grids - one for before and one for after and temp points to "before" while next points to "after"
float* tgrid;
MPI_Request send_req, recv_req;										// MPI requests for wait
MPI_Status send_stat, recv_stat;									// MPI statuses for wait
MPI_Comm cartesian;													// New communicator
MPI_Datatype row;													// MPI vector datatype for rows
MPI_Datatype column;												// MPI vector datatype for columns

// initialize MPI
MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

// check multithread safety
if (provided < MPI_THREAD_MULTIPLE)
{
printf("Requested thread level was unavailable\n");
MPI_Finalize();
exit(EXIT_FAILURE);
}

// arguments count
if (argc != 3)
{
printf("Usage: heat <grid_blocks> <time_steps>\n");
MPI_Finalize();
exit(EXIT_FAILURE);
}

// arguments check
if (arg_check(argv, numprocs, &N, &BLOCKS, &STEPS, &DIM) < 0)
{
printf("Arguments must be positive integers with first argument being an integer multiple of the number of processes\n");
MPI_Finalize();
exit(EXIT_FAILURE);
}

// topology parameters initialization
mpi_dim[0] = sqrt(numprocs);
mpi_dim[1] = mpi_dim[0];
period[0] = 1;
period[1] = 1;
reorder = 1;

// 2D topology create
MPI_Cart_create(MPI_COMM_WORLD, 2, mpi_dim, period, reorder, &cartesian);

// Topology neighbors
MPI_Cart_shift(cartesian, 0, 1, &left, &right);
MPI_Cart_shift(cartesian, 1, 1, &top, &bottom);

// datatypes initialize
MPI_Type_vector(DIM, 1, DIM + 2, MPI_FLOAT, &column);
MPI_Type_commit(&column);
MPI_Type_vector(DIM, 1, 1, MPI_FLOAT, &row);
MPI_Type_commit(&row);

// grids create
if ((fgrid = grid_create(DIM + 2)) == NULL)
{
perror("grid_create: ");
exit(EXIT_FAILURE);
}
if ((lgrid = grid_create(DIM + 2)) == NULL)
{
perror("grid_create: ");
exit(EXIT_FAILURE);
}

// start grid initialize
grid_init(fgrid, DIM + 2, myrank);

// start grid print (for debugging purposes)
//if (myrank == 0)
//	grid_print(fgrid, DIM + 2, "initial.dat");

cur_grid = 0;

// set "before" and "after" grid
temp_grid = (cur_grid == 0 ? fgrid : lgrid);
next_grid = (cur_grid != 0 ? fgrid : lgrid);

// wait for all hosts to be initialized - synchronize
MPI_Barrier(cartesian);

// define start time for update
if (myrank == 0)
start_time = MPI_Wtime();

if (numprocs == 1)
{
#		pragma omp parallel num_threads(4) private(i, j, counter)
{
for (counter = 0; counter < STEPS; counter++)
{
#				pragma omp for schedule(static) collapse(2)
for (i = 1; i <= DIM; i++)
{
for (j = 1; j <= DIM; j++)
{
next_grid[i * (DIM + 2) + j] = temp_grid[i * (DIM + 2) + j] +
parms.cx * ((temp_grid[(i + 1) * (DIM + 2) + j]) +
temp_grid[(i - 1) * (DIM + 2) + j] - 2.0 * temp_grid[i * (DIM + 2) + j]) +
parms.cy * (temp_grid[i * (DIM + 2) + (j + 1)] +
temp_grid[i * (DIM + 2) + (j - 1)] - 2.0 * temp_grid[i * (DIM + 2) + j]);
}
}
#				pragma omp master
{
if (counter != STEPS - 1)
{	
cur_grid = 1 - cur_grid;

// set "before" and "after" grid
temp_grid = (cur_grid == 0 ? fgrid : lgrid);
next_grid = (cur_grid != 0 ? fgrid : lgrid);
}
}
#				pragma omp barrier
}
}
}
else
{
#		pragma omp parallel num_threads(4) private(i, j, k, l, send_req, recv_req, send_stat, recv_stat, counter)
{
int mrank;												// Rank of thread

mrank = omp_get_thread_num();

for (counter = 0; counter < STEPS; counter++)
{
if (mrank == 0)
{
// sends left
MPI_Isend(&temp_grid[1 * (DIM + 2) + 1], 1, column, left, mrank, cartesian, &send_req);

// receives left
MPI_Irecv(&temp_grid[1 * (DIM + 2) + 0], 1, column, left, mrank + 1, cartesian, &recv_req);
}
else if (mrank == 1)
{
// sends right
MPI_Isend(&temp_grid[1 * (DIM + 2) + DIM], 1, column, right, mrank, cartesian, &send_req);

// receives right
MPI_Irecv(&temp_grid[1 * (DIM + 2) + (DIM + 1)], 1, column, right, mrank - 1, cartesian, &recv_req);
}
else if (mrank == 2)
{
// sends top
MPI_Isend(&temp_grid[1 * (DIM + 2) + 1], 1, row, top, mrank, cartesian, &send_req);

// receives top
MPI_Irecv(&temp_grid[0 * (DIM + 2) + 1], 1, row, top, mrank + 1, cartesian, &recv_req);
}
else
{
// sends bottom
MPI_Isend(&temp_grid[DIM * (DIM + 2) + 1], 1, row, bottom, mrank, cartesian, &send_req);

// receives bottom
MPI_Irecv(&temp_grid[(DIM + 1) * (DIM + 2) + 1], 1, row, bottom, mrank - 1, cartesian, &recv_req);
}

#				pragma omp for schedule(static) collapse(2)
for (i = 2; i < DIM; i++)
{
for (j = 2; j < DIM; j++)
{
next_grid[i * (DIM + 2) + j] = temp_grid[i * (DIM + 2) + j] +
parms.cx * ((temp_grid[(i + 1) * (DIM + 2) + j]) +
temp_grid[(i - 1) * (DIM + 2) + j] - 2.0 * temp_grid[i * (DIM + 2) + j]) +
parms.cy * (temp_grid[i * (DIM + 2) + (j + 1)] +
temp_grid[i * (DIM + 2) + (j - 1)] - 2.0 * temp_grid[i * (DIM + 2) + j]);
}
}

if (mrank == 0)
{
// wait for incoming request
MPI_Wait(&recv_req, &recv_stat);

#					pragma omp barrier

// update left border
l = 1;
for (k = 2; k < DIM; k++) 
next_grid[k * (DIM + 2) + l] = temp_grid[k * (DIM + 2) + l] +
parms.cx * ((temp_grid[(k + 1) * (DIM + 2) + l]) +
temp_grid[(k - 1) * (DIM + 2) + l] - 2.0 * temp_grid[k * (DIM + 2) + l]) +
parms.cy * (temp_grid[k * (DIM + 2) + (l + 1)] +
temp_grid[k * (DIM + 2) + (l - 1)] - 2.0 * temp_grid[k * (DIM + 2) + l]);

// wait for outgoing request
MPI_Wait(&send_req, &send_stat);
}
else if (mrank == 1)
{
// wait for incoming request
MPI_Wait(&recv_req, &recv_stat);

#					pragma omp barrier

// update right border
l = DIM;
for (k = 2; k < DIM; k++) 
next_grid[k * (DIM + 2) + l] = temp_grid[k * (DIM + 2) + l] +
parms.cx * ((temp_grid[(k + 1) * (DIM + 2) + l]) +
temp_grid[(k - 1) * (DIM + 2) + l] - 2.0 * temp_grid[k * (DIM + 2) + l]) +
parms.cy * (temp_grid[k * (DIM + 2) + (l + 1)] +
temp_grid[k * (DIM + 2) + (l - 1)] - 2.0 * temp_grid[k * (DIM + 2) + l]);

// wait for outgoing request
MPI_Wait(&send_req, &send_stat);
}
else if (mrank == 2)
{
// wait for incoming request
MPI_Wait(&recv_req, &recv_stat);

#					pragma omp barrier

// update top border
k = 1;
for (l = 1; l < DIM + 1; l++) 
next_grid[k * (DIM + 2) + l] = temp_grid[k * (DIM + 2) + l] +
parms.cx * ((temp_grid[(k + 1) * (DIM + 2) + l]) +
temp_grid[(k - 1) * (DIM + 2) + l] - 2.0 * temp_grid[k * (DIM + 2) + l]) +
parms.cy * (temp_grid[k * (DIM + 2) + (l + 1)] +
temp_grid[k * (DIM + 2) + (l - 1)] - 2.0 * temp_grid[k * (DIM + 2) + l]);

// wait for outgoing request
MPI_Wait(&send_req, &send_stat);
}
else
{
// wait for incoming request
MPI_Wait(&recv_req, &recv_stat);

#					pragma omp barrier

// update bottom border
k = DIM;
for (l = 1; l < DIM + 1; l++) 
next_grid[k * (DIM + 2) + l] = temp_grid[k * (DIM + 2) + l] +
parms.cx * ((temp_grid[(k + 1) * (DIM + 2) + l]) +
temp_grid[(k - 1) * (DIM + 2) + l] - 2.0 * temp_grid[k * (DIM + 2) + l]) +
parms.cy * (temp_grid[k * (DIM + 2) + (l + 1)] +
temp_grid[k * (DIM + 2) + (l - 1)] - 2.0 * temp_grid[k * (DIM + 2) + l]);

// wait for outgoing request
MPI_Wait(&send_req, &send_stat);
}

#				pragma omp barrier
#				pragma omp master
{
if (counter != STEPS - 1)
{	
cur_grid = 1 - cur_grid;

// set "before" and "after" grid
temp_grid = (cur_grid == 0 ? fgrid : lgrid);
next_grid = (cur_grid != 0 ? fgrid : lgrid);
}
}
#				pragma omp barrier
}
}
}

// calculate elapsed time for update
if (myrank == 0)
{
end_time = MPI_Wtime();

MPI_Get_processor_name(processor_name, &namelen);
printf("Process 0 running on %s finished in %f seconds\n", processor_name, end_time - start_time);

// final grid print (for debugging purposes)
//grid_print(next_grid, DIM + 2, "final.dat");
}

// free grids
grid_destr(fgrid, DIM + 2);
grid_destr(lgrid, DIM + 2);

// free datatypes
MPI_Type_free(&row);
MPI_Type_free(&column);

MPI_Finalize();

exit(EXIT_SUCCESS);
}
