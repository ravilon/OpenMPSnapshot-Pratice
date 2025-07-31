/**
* Author: https://www.dreamincode.net/forums/topic/336580-recursive-algorithm-for-n-queens-problem/
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

unsigned int solutions;

void setQueen(int queens[], int row, int col, int size) 
{
//check all previously placed rows for attacks
for(int i = 0; i < row; i++) {
// vertical clashes
if (queens[i] == col) {
return;
}
// diagonal clashes
if (abs(queens[i] - col) == (row - i)) {
return;
}
}
// no clashes found is ok, set the queen
queens[row] = col;

// if we're at the end of the rows
if(row == size - 1) {
#pragma omp atomic
solutions++;  // found a solution
}
// else we'll try to fill next row
else {
for(int i = 0; i < size; i++) {
setQueen(queens, row + 1, i, size);
}
}
}

// function to find the solutions
void solve(int size) 
{
for(int i = 0; i < size; i++) {
// array representing queens placed on a chess board. Index is row, value is column.
int *queens = malloc(sizeof(int)*size); 
setQueen(queens, 0, i, size);
free(queens);
}
}

int main(int argc, char* argv[])
{
double start_time, end_time;

if (argc != 2){
printf("ERROR! Usage: ./executable size\n");
return EXIT_FAILURE;
}

int size = atoi(argv[1]);

start_time = omp_get_wtime();

solve(size);

// get end time
end_time = omp_get_wtime();
// print results
printf("Sequential Solution with a size of n = %d\n", size);
printf("The execution time is %g sec\n", end_time - start_time);
printf("Number of found solutions is %d\n", solutions);

return EXIT_SUCCESS;
}