/****************************************************************************
 *
 * my-mandelbrot.c - saves the Mandelbrot set into a CSV file
 * 
 * Credits: Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * Copyright (C) 2022 by Matteo Barbetti <matteo.barbetti(at)unifi.it>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * --------------------------------------------------------------------------
 *
 * This program computes the Mandelbrot set and exports it for visualization. 
 * In this script, the inner for loop (lines 94-102) of the nested ones that 
 * compute the Mandelbrot set is parallelized. A "runtime" scheduling clause is 
 * used, then you are free to decide the preferred scheduling strategy at runtime.
 *
 * Compile with:
 * gcc -std=c99 -Wall -Wpedantic -fopenmp my-mandelbrot.c -o my-mandelbrot.out
 *
 * Run with:
 * OMP_NUM_THREADS=4 OMP_SCHEDULE="static,1" ./my-mandelbrot.out [x_size y_size]
 * 
 * (x_size, y_size) = picture window in pixels; default (1024, 768)
 *
 ****************************************************************************/

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>

const int MAXIT = 10000;

/* Fills m x n matrix M with zero values */
void fill( int* M, int m, int n )
{
    int i, j;
    for ( i = 0; i < m; i++ ) {
        for ( j = 0; j < n; j++ ) {
            M[i*m + j] = (int)(0);
        }
    }
}

/*
 * Iterates the recurrence:
 *
 * z_0 = 0;
 * z_{n+1} = z_n^2 + (cx + i*cy);
 *
 * Returns the first n such that ||z_n|| > 2, or |MAXIT|
 */
int iterate( float cx, float cy )
{
    float x = 0.0f, y = 0.0f, xnew, ynew;
    int it;
    for ( it = 0; (it < MAXIT) && (x*x + y*y <= 2.0*2.0); it++ ) {
        xnew = x*x - y*y + cx;
        ynew = 2.0*x*y + cy;
        x = xnew;
        y = ynew;
    }
    return it;
}

int main( int argc, char *argv[] )
{
    int x, y;
    int *matrix;

    /* Picture window size, in pixels */
    int x_size = 1024, y_size = 768;

    if ( argc > 3 ) {
        printf("Usage: %s [x_size y_size]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc == 3 ) {
        x_size = atoi(argv[1]);
        y_size = atoi(argv[2]);
    }

    /* Coordinates of the bounding box of the Mandelbrot set */
    const double x_min = -2.5, x_max = 1.5;
    const double y_min = -1.5, y_max = 1.5;

    matrix = (int*) malloc( y_size * x_size * sizeof(int) );
    fill(matrix, y_size, x_size);

    const double tstart = hpc_gettime(); 
#if __GNUC__ < 9
#pragma omp parallel for private(x) schedule(runtime)
#else
#pragma omp parallel for private(x) shared(x_size,x_min,x_max,y_size,y_min,y_max,matrix,MAXIT) schedule(runtime)
#endif
    for ( y = 0; y < y_size; y++ ) {
	for ( x = 0; x < x_size; x++ ) {
            const double re = x_min + (x_max - x_min) * (float)(x) / (x_size - 1);
            const double im = y_max - (y_max - y_min) * (float)(y) / (y_size - 1);
            const int it = iterate(re, im);
#pragma omp critical
	    if ( it < MAXIT ) {
	        matrix[y*y_size + x] = it;
	    }
	}
    }
    const double elapsed = hpc_gettime() - tstart;
    printf ("Elapsed time: %f\n", elapsed);

    char filepath[256];
    snprintf ( filepath, sizeof(filepath), "./data/mandelbrot/matrix_%dx%d.csv", x_size, y_size );
    FILE *fpt = fopen ( filepath, "w" );
    for ( y = 0; y < y_size; y++ ) {
        for ( x = 0; x < x_size; x++ ) {
	    if ( x == x_size - 1 ) {
	        fprintf ( fpt, "%d\n", matrix[y*y_size + x] );
	    } else {
	        fprintf ( fpt, "%d,", matrix[y*y_size + x] );
	    }
	}
    }
    fclose(fpt);
    // printf ("Mandelbrot matrix correctly exported to %s\n", filepath);

    return EXIT_SUCCESS;
}
