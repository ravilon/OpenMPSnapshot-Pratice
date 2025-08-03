#include "game-of-life.hpp"

int GameOfLife::populateCurrentWorld() {
population = 0;
if (game == DEFAULT) {
// randomly generated
for (int i = 1; i < nx - 1; i++) {
for (int j = 1; j < ny - 1; j++) {
if (real_rand() < prob) {
currWorld[i][j] = true;
population++;
} else {
currWorld[i][j] = false;
}
}
}

} else if (game == BLOCK) {
// still block-life
printf("2x2 Block, still life\n");
int nx2 = nx / 2;
int ny2 = ny / 2;
currWorld[nx2 + 1][ny2 + 1] = currWorld[nx2][ny2 + 1] = currWorld[nx2 + 1][ny2] = currWorld[nx2][ny2] = 1;
population = 4;
}
// @note can add more games here
return population;
}

void GameOfLife::serial() {
for (int iter = 0; iter < maxiter && population != 0; ++iter) {
population = 0;
for (int i = 1; i < nx - 1; ++i) {
for (int j = 1; j < ny - 1; ++j) {
// calculate neighbor count
int nn = currWorld[i + 1][j] + currWorld[i - 1][j] + currWorld[i][j + 1] + currWorld[i][j - 1] +
currWorld[i + 1][j + 1] + currWorld[i - 1][j - 1] + currWorld[i - 1][j + 1] + currWorld[i + 1][j - 1];
// if alive check if you die, if dead check if you can produce.
nextWorld[i][j] = currWorld[i][j] ? (nn == 2 || nn == 3) : (nn == 3);
// update population
population += nextWorld[i][j];
}
}

// swap pointers
tmpWorld = nextWorld;
nextWorld = currWorld;
currWorld = tmpWorld;

// plot
if (plotter) plotter->plot(iter, population, currWorld);
}
}

void GameOfLife::parallel() {
int i, j, iter;
int sum = 0;

for (iter = 0; iter < maxiter && population; ++iter) {
population = 0;
sum = 0;

// spawn 2 threads
#pragma omp parallel num_threads(2) if (numthreads > 1)
{
#pragma omp single
{
// task 1: plotting
#pragma omp task
{
if (plotter) plotter->plot(iter, population, currWorld);
}  // end of plot task

// task 2: computing
#pragma omp task
{
// spawn rest of the threads
#pragma omp parallel num_threads(numthreads - 1) if (numthreads > 2)
{
// a sum reduction in parallel
#pragma omp for reduction(+ : sum) private(j)
for (i = 1; i < nx - 1; i++) {
// we could use collapse(2) but then it would be 2D decomposition, we decided 1D only
for (j = 1; j < ny - 1; j++) {
//  calculate neighbor count
int nn = currWorld[i + 1][j] + currWorld[i - 1][j] + currWorld[i][j + 1] + currWorld[i][j - 1] +
currWorld[i + 1][j + 1] + currWorld[i - 1][j - 1] + currWorld[i - 1][j + 1] +
currWorld[i + 1][j - 1];
// if alive check if you die, if dead check if you can produce.
nextWorld[i][j] = currWorld[i][j] ? (nn == 2 || nn == 3) : (nn == 3);
// update population (CRITICAL)
sum += nextWorld[i][j];
}
}  // end of for reduction

#pragma omp single nowait
{ population += sum; }  // end of single nowait
}                         // end of parallel N-1 threads
}                           // end of compute task
}                             // end of single
}                               // end of parallel 2 threads

// pointer swap
tmpWorld = nextWorld;
nextWorld = currWorld;
currWorld = tmpWorld;
}

// we have one more print, because this was calculated at the last iteration
if (plotter) plotter->plot(iter, population, currWorld);
}

GameOfLife::GameOfLife(int nx, int ny, int numthreads, int maxiter, float prob, bool isPlotEnabled, game_e game) {
// add ghost regions
nx += 2;
ny += 2;

// update parameters
this->nx = nx;
this->ny = ny;
this->numthreads = numthreads;
this->maxiter = maxiter;
this->prob = prob;
this->isPlotEnabled = isPlotEnabled;
this->game = game;

// allocate current world (which you read from)
int i;
currWorld = (bool **)malloc(sizeof(bool *) * nx + sizeof(bool) * nx * ny);
for (i = 0; i < nx; i++) currWorld[i] = (bool *)(currWorld + nx) + i * ny;

// allocate next world (which you write to)
nextWorld = (bool **)malloc(sizeof(bool *) * nx + sizeof(bool) * nx * ny);
for (i = 0; i < nx; i++) nextWorld[i] = (bool *)(nextWorld + nx) + i * ny;

// reset boundaries
for (i = 0; i < nx; i++) {
currWorld[i][0] = 0;
currWorld[i][ny - 1] = 0;
nextWorld[i][0] = 0;
nextWorld[i][ny - 1] = 0;
}
for (i = 0; i < ny; i++) {
currWorld[0][i] = 0;
currWorld[nx - 1][i] = 0;
nextWorld[0][i] = 0;
nextWorld[nx - 1][i] = 0;
}
this->populateCurrentWorld();

// prepare plotter
if (isPlotEnabled) plotter = new GameOfLifePlotter(nx, ny);
}

GameOfLife::~GameOfLife() {
// frees
if (isPlotEnabled) delete plotter;
free(nextWorld);
free(currWorld);
}

void GameOfLife::printParameters(runtype_e runType) {
printf(
"%s\n\t"
"Probability: %f\n\t"
"Threads: %d\n\t"
"Iterations: %d\n\t"
"Problem Size: %d x %d\n",
runType == SERIAL ? "Serial" : "Parallel", prob, numthreads, maxiter, nx, ny);
}