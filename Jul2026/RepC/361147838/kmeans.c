#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

//***********************************************************************************************
/*
Demo code - k means algorith implementation

Classify vectors based on their attributes in a number of clusters. The vectors are initialized
randomly and the number of vectors, attributes and centers are controlled before compilation.
*/
//***********************************************************************************************

#define N 100000            // number of vectors
#define Nv 1000             // number of values in each vector (dimensions)
#define Nc 100              // number of centers (clusters)
#define THR_KMEANS 0.000001 // kmeans threshold

// ***********************************************************************************************

void createVectors();
void initCenters();
void classification();
void estimateCenters(void);
int terminate(void);

// debugging functions
void print2dFloatMatrix(float *mat, int x, int y);
void print1dIntMatrix(int *mat, int x);

// ***********************************************************************************************

float vectors[N][Nv];
float centers[Nc][Nv];
int classes[N]; // each vector is assigned to a center (cluster)
// define the previous iteration distance sum and current iteration distance sum
// they are used by terminate() to determine the end of the algorithm
float prevDistSum;
float currentDistSum;
float dist;
float distmin;
int indexmin;
int vecsInCenter;
float newCenterValues[Nv];

int main()
{
createVectors();
// print2dFloatMatrix((float *)vectors, N, Nv);

initCenters();
// print2dFloatMatrix((float *)centers, Nc, Nv);

do
{
classification();
// print1dIntMatrix((int *)classes, N);
// putchar('\n');
estimateCenters();
} while (terminate());
// print2dFloatMatrix((float *)centers, Nc, Nv);
// printf("%d %f\n", x, sumDifference);
return 0;
}

// ***********************************************************************************************

// fill vectors[] with N random floats between [0.0, 1.0]
void createVectors()
{
for (int i = 0; i < N; i++)
for (int j = 0; j < Nv; j++)
vectors[i][j] = (float)rand() / (float)RAND_MAX;
}

// ***********************************************************************************************

// choose Nc different vectors from vectors[] and move them into centers[]
void initCenters()
{
// the first Nc vectors are used as centers because of the way they are generated (randomly)
memcpy(centers, vectors, sizeof(centers));
}

// ***********************************************************************************************

void classification()
{
int i, j, k;
prevDistSum = currentDistSum;   // overwrite the previous sum of distances
currentDistSum = 0;             // initialize the current sum of distances
// parallelization of the evaluation of the center for every point
#pragma omp parallel for private(i, j, k, dist, distmin, indexmin) reduction(+:currentDistSum) schedule(static)
for (i = 0; i < N; i++)
{
distmin = 0;        
indexmin = 0;               // initialize the values of indexmin and distmin
for (k = 0; k < Nv; k++)    // as if the closest center is center[0]
{
distmin += (vectors[i][k] - centers[0][k]) * (vectors[i][k] - centers[0][k]);
}
for (j = 1; j < Nc; j++)
{
dist = 0;
// indicate to the omp preprocessor that the following for loop will have vector processing optimizations
#pragma omp simd
for (int k = 0; k < Nv; k++)
dist += (vectors[i][k] - centers[j][k]) * (vectors[i][k] - centers[j][k]);
// dist is now the distance to the center[j]
if (dist < distmin)
{
distmin = dist; // store minimum distance and index of the new center
indexmin = j;
}
}
classes[i] = indexmin; // after iterating over all centers, we now know the closest
currentDistSum += distmin;
}
}

// ***********************************************************************************************

// calculate a new center as the mean value of the vectors of each former center
void estimateCenters(void)
{
for (int i = 0; i < Nc; i++) // iterate over centers
{
vecsInCenter = 0;
memset(newCenterValues, 0, sizeof(newCenterValues));
for (int j = 0; j < N; j++) // iterate over all vectors
if (classes[j] == i) // continue only with the vectors classified with the current center
{
vecsInCenter++;
for (int k = 0; k < Nv; k++)
newCenterValues[k] += vectors[j][k];
}
for (int k = 0; k < Nv; k++)
// iterate over the new center values and take the mean value of each one
centers[i][k] = newCenterValues[k] / vecsInCenter;
}
}

// ***********************************************************************************************

int x = 0;
int terminate(void)
{
// float sumDifference = fabs(prevDistSum - currentDistSum);
// if (sumDifference < THR_KMEANS)
//     return 0;
printf("%f\n", currentDistSum);
if (x == 10)
return 0;
x++;
return 1;
}

// ***********************************************************************************************

void print2dFloatMatrix(float *mat, int x, int y)
{
for (int i = 0; i < x; i++)
{
printf("\n%d\n", i);
for (int j = 0; j < y; j++)
printf("%f ", *(mat + (y * i) + j));
}
printf("\n");
}

void print1dIntMatrix(int *mat, int x)
{
for (int x = 0; x < N; x++)
printf("%d\n", mat[x]);
}