#include <omp.h>
#include <stdio.h>
#include <string.h>
#include "timestamp.h"

// parameters for heat transfer
struct Parms
{ 
	float cx;
	float cy;
} parms = {0.1, 0.1};

// updates the whole grid
float update_cpu(float* dest, float* source, const int dim, const int steps)
{
	float* temp_grid;													// Two grids - one for before and one for after and temp points to "before"
	float* next_grid;													// next points to "after"
	int cur_grid;														// IF 0 -> source is current temperature grid ELSE dest ... 
	int i, j, counter;
	timestamp s_time;
	float t_time;
	
	cur_grid = 0;
	
	// define start time for update
	s_time = getTimestamp();
	
	
	// set "before" and "after" grid
	temp_grid = (cur_grid == 0 ? source : dest);
	next_grid = (cur_grid != 0 ? source : dest);
	
	// create 4 threads and split for to these (default by openmp)
#	pragma omp parallel num_threads(4) private(i, j, counter)
	{
		for (counter = 0; counter < steps; counter++)
		{
#			pragma omp for schedule(static) collapse(2)
				for (i = 1; i <= dim; i++)
					for (j = 1; j <= dim; j++)
						next_grid[i * (dim + 2) + j] = temp_grid[i * (dim + 2) + j] +
							parms.cx * ((temp_grid[(i + 1) * (dim + 2) + j]) +
								temp_grid[(i - 1) * (dim + 2) + j] - 2.0 * temp_grid[i * (dim + 2) + j]) +
							parms.cy * (temp_grid[i * (dim + 2) + (j + 1)] +
								temp_grid[i * (dim + 2) + (j - 1)] - 2.0 * temp_grid[i * (dim + 2) + j]);
		}
#		pragma omp master
		{
			if (counter != steps - 1)
			{	
				cur_grid = 1 - cur_grid;
				
				// set "before" and "after" grid
				temp_grid = (cur_grid == 0 ? source : dest);
				next_grid = (cur_grid != 0 ? source : dest);
			}
		}
#		pragma omp barrier
	}
	
	// calculate elapsed time
	t_time = getElapsedtime(s_time);
	
	// make sure that destination grid is updated
	if (dest != next_grid)
		memcpy(dest, next_grid, (dim + 2) * (dim + 2) * sizeof(float));
	
	return t_time;
}
