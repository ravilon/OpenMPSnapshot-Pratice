#include <stdio.h>
#include <stdlib.h>
#include <omp.h>



/*Create a 2D array with contiguous memory*/
char **allocateArray(int n, int init_flag)
{
int  i, j;
char *p, **array;

p = malloc(n*n*sizeof(char));		//Allocate an 1D array of n*n contiguous chunks
array = malloc(n*sizeof(char*));
for (i = 0; i < n; i++){			//Allocate a pointers array that points every row of the above 1D array
array[i] = &(p[i*n]);			//Just like a usual dynamically allocated 2D array
for (j = 0; j < n; j++){
if (init_flag == 0) array[i][j] = 0;			//Initialize the array items as 0s
else array[i][j] = rand() % 2;
}
}
return array;
}



/*Delete a 2D array created in the way above*/
void deleteArray(char ***array)
{
free(&((*array)[0][0]));			//Delete the initial 1D array with n*n items
free(*array);						//Delete the pointers array (that shows every "row")
}



/*Print the cells array in command line*/
void show(char **array, int N)
{
int i, j;
printf("\n///////////////////////////////////////////////////\n\n");
for (i = 0; i < N; i++){
for (j = 0; j < N; j++){
if (array[i][j] == 0) printf("-");
else if (array[i][j] == 1) printf("X");
else printf("?");
}
printf("\n");
}
}



/*The side cells evolve - move to next generation*/
void evolve_sides(char **old_gen, char **new_gen, char **border, int N, int thread_count, int *allzeros, int *change)
{
int i, j, neighbors;

/*Calculate the neighbors of cells in lines based on received_border array*/
#	pragma omp parallel num_threads(thread_count) \
default(none) shared(old_gen, new_gen, border, N, allzeros, change) private(j, neighbors)

for (j = 0; j < N; j++)
{															//Up line
neighbors = 0;
if ((j == 0) && (border[4][0] == 1)) neighbors++;				//Upleft
else if ((j != 0) && (border[5][j-1] == 1)) neighbors++;
if (border[5][j] == 1) neighbors++;								//Up
if ((j == N-1) && (border[6][0] == 1)) neighbors++;				//Upright
else if ((j != N-1) && (border[5][j+1] == 1)) neighbors++;
if ((j == N-1) && (border[7][0] == 1)) neighbors++;				//Right
else if ((j != N-1) && (old_gen[0][j+1] == 1)) neighbors++;
if ((j == N-1) && (border[7][1] == 1)) neighbors++;				//Downright
else if ((j != N-1) && (old_gen[1][j+1] == 1)) neighbors++;
if (old_gen[1][j] == 1) neighbors++;							//Down
if ((j == 0) && (border[3][1] == 1)) neighbors++;				//Downleft
else if ((j != 0) && (old_gen[1][j-1] == 1)) neighbors++;
if ((j == 0) && (border[3][0] == 1)) neighbors++;				//Left
else if ((j != 0) && (old_gen[0][j-1] == 1)) neighbors++;
/*--------------------------------------------------------------------------*/
if ((old_gen[0][j] == 1) && ((neighbors < 2) || (neighbors > 3)))
new_gen[0][j] = 0;
else if ((old_gen[0][j] == 0) && (neighbors == 3))
new_gen[0][j] = 1;
else
new_gen[0][j] = old_gen[0][j];		//Assign new values in up line's cells (if alive or dead)

if (new_gen[0][j] != 0) *allzeros = 1;
if (new_gen[0][j] != old_gen[0][j]) *change = 1;

//Down line
neighbors = 0;
if ((j == 0) && (border[3][N-2] == 1)) neighbors++;				//Upleft
else if ((j != 0) && (old_gen[N-2][j-1] == 1)) neighbors++;
if (old_gen[N-2][j] == 1) neighbors++;							//Up
if ((j == N-1) && (border[7][N-2] == 1)) neighbors++;			//Upright
else if ((j != N-1) && (old_gen[N-2][j+1] == 1)) neighbors++;
if ((j == N-1) && (border[7][0] == 1)) neighbors++;				//Right
else if ((j != N-1) && (old_gen[N-1][j+1] == 1)) neighbors++;
if ((j == N-1) && (border[0][0] == 1)) neighbors++;				//Downright
else if ((j != N-1) && (border[1][j+1] == 1)) neighbors++;
if (border[1][j] == 1) neighbors++;								//Down
if ((j == 0) && (border[2][0] == 1)) neighbors++;				//Downleft
else if ((j != 0) && (border[1][j-1] == 1)) neighbors++;
if ((j == 0) && (border[3][0] == 1)) neighbors++;				//Left
else if ((j != 0) && (old_gen[N-1][j-1] == 1)) neighbors++;
/*--------------------------------------------------------------------------*/
if ((old_gen[N-1][j] == 1) && ((neighbors < 2) || (neighbors > 3)))
new_gen[N-1][j] = 0;
else if ((old_gen[N-1][j] == 0) && (neighbors == 3))
new_gen[N-1][j] = 1;
else
new_gen[N-1][j] = old_gen[N-1][j];		//Assign new values in down line's cells (if alive or dead)

if (new_gen[N-1][j] != 0) *allzeros = 1;
if (new_gen[N-1][j] != old_gen[N-1][j]) *change = 1;
}

/*Calculate the neighbors of cells in rows based on received_border array*/
#	pragma omp parallel num_threads(thread_count) \
default(none) shared(old_gen, new_gen, border, N, allzeros, change) private(i, neighbors)

for (i = 0; i < N; i++)
{															//Left line
neighbors = 0;
if ((i == 0) && (border[4][0] == 1)) neighbors++;				//Upleft
else if ((i != 0) && (border[3][i-1] == 1)) neighbors++;
if ((i == 0) && (border[5][0] == 1)) neighbors++;				//Up
else if ((i != 0) && (old_gen[i-1][0] == 1)) neighbors++;
if ((i == 0) && (border[5][1] == 1)) neighbors++;				//Upright
else if ((i != 0) && (old_gen[i-1][1] == 1)) neighbors++;
if (old_gen[i][1] == 1) neighbors++;							//Right
if ((i == N-1) && (border[1][1] == 1)) neighbors++;				//Downright
else if ((i != N-1) && (old_gen[i+1][1] == 1)) neighbors++;
if ((i == N-1) && (border[1][0] == 1)) neighbors++;				//Down
else if ((i != N-1) && (old_gen[i+1][0] == 1)) neighbors++;
if ((i == N-1) && (border[2][0] == 1)) neighbors++;				//Downleft
else if ((i != N-1) && (border[3][i+1] == 1)) neighbors++;
if (border[3][i] == 1) neighbors++;								//Left
/*--------------------------------------------------------------------------*/
if ((old_gen[i][0] == 1) && ((neighbors < 2) || (neighbors > 3)))
new_gen[i][0] = 0;
else if ((old_gen[i][0] == 0) && (neighbors == 3))
new_gen[i][0] = 1;
else
new_gen[i][0] = old_gen[i][0];		//Assign new values in left row's cells (if alive or dead)

if (new_gen[i][0] != 0) *allzeros = 1;
if (new_gen[i][0] != old_gen[i][0]) *change = 1;

//Right line
neighbors = 0;
if ((i == 0) && (border[5][N-2] == 1)) neighbors++;				//Upleft
else if ((i != 0) && (old_gen[i-1][N-2] == 1)) neighbors++;
if ((i == 0) && (border[5][N-1] == 1)) neighbors++;				//Up
else if ((i != 0) && (old_gen[i-1][N-1] == 1)) neighbors++;
if ((i == 0) && (border[6][0] == 1)) neighbors++;				//Upright
else if ((i != 0) && (border[7][i-1] == 1)) neighbors++;
if (border[7][i] == 1) neighbors++;								//Right
if ((i == N-1) && (border[0][0] == 1)) neighbors++;				//Downright
else if ((i != N-1) && (border[7][i+1] == 1)) neighbors++;
if ((i == N-1) && (border[1][N-1] == 1)) neighbors++;			//Down
else if ((i != N-1) && (old_gen[i+1][N-1] == 1)) neighbors++;
if ((i == N-1) && (border[1][N-2] == 1)) neighbors++;			//Downleft
else if ((i != N-1) && (old_gen[i+1][N-2] == 1)) neighbors++;
if (old_gen[i][N-2] == 1) neighbors++;							//Left
/*--------------------------------------------------------------------------*/
if ((old_gen[i][N-1] == 1) && ((neighbors < 2) || (neighbors > 3)))
new_gen[i][N-1] = 0;
else if ((old_gen[i][N-1] == 0) && (neighbors == 3))
new_gen[i][N-1] = 1;
else
new_gen[i][N-1] = old_gen[i][N-1];		//Assign new values in right row's cells (if alive or dead)

if (new_gen[i][N-1] != 0) *allzeros = 1;
if (new_gen[i][N-1] != old_gen[i][N-1]) *change = 1;
}
}



/*The inner cells evolve (not the side ones)*/
void evolve_inner(char **old_gen, char **new_gen, int N, int thread_count, int *allzeros, int *change)
{
int i, j, up, down, left, right, neighbors;

/*Calculate the "inside" cells, from (1,1) till (N-1,N-1)*/
#	pragma omp parallel num_threads(thread_count) \
default(none) shared(old_gen, new_gen, N, allzeros, change) private(i, j, up, down, left, right, neighbors)

for (i = 1; i < N-1; i++)
{										//Calculate up and down rows
up   = i-1;
down = i+1;

#	pragma omp for
for (j = 1; j < N-1; j++)
{									//Calculate left and right columns
left  = j-1;
right = j+1;
neighbors = 0;

if (old_gen[up][left]    == 1) neighbors++;				//Check the neighbors
if (old_gen[up][j]       == 1) neighbors++;
if (old_gen[up][right]   == 1) neighbors++;
if (old_gen[i][right]    == 1) neighbors++;
if (old_gen[down][right] == 1) neighbors++;
if (old_gen[down][j]     == 1) neighbors++;
if (old_gen[down][left]  == 1) neighbors++;
if (old_gen[i][left]     == 1) neighbors++;

if ((old_gen[i][j] == 1) && ((neighbors < 2) || (neighbors > 3)))
new_gen[i][j] = 0;
else if ((old_gen[i][j] == 0) && (neighbors == 3))
new_gen[i][j] = 1;
else
new_gen[i][j] = old_gen[i][j];		//Assign new values in "inside" cells (if alive or dead)

if (new_gen[i][j] != 0) *allzeros = 1;
if (new_gen[i][j] != old_gen[i][j]) *change = 1;
}
}
}