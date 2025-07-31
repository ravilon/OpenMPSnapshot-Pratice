
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <omp.h>
#include <likwid.h>


int main(int argc, char* argv[])
{
int i, j;
int err;
int* cpus;
int gid;
double result = 0.0;
char estr[] = "PWR_PKG_ENERGY:PWR0";
char* enames[1] ={"PWR_PKG_ENERGY:PWR0"};
int n = sizeof(enames) / sizeof(enames[0]);

//perfmon_setVerbosity(3);
// Load the topology module and print some values.
err = topology_init();
if (err < 0)
{
printf("Failed to initialize LIKWID's topology module\n");
return 1;
}
// CpuInfo_t contains global information like name, CPU family, ...
CpuInfo_t info = get_cpuInfo();
// CpuTopology_t contains information about the topology of the CPUs.
CpuTopology_t topo = get_cpuTopology();
// Create affinity domains. Commonly only needed when reading Uncore counters
affinity_init();

printf("Likwid example on a %s with %d CPUs\n", info->name, topo->numHWThreads);

cpus = (int*)malloc(topo->numHWThreads * sizeof(int));
if (!cpus)
return 1;

for (i=0;i<topo->numHWThreads;i++)
{
cpus[i] = topo->threadPool[i].apicId;
}

// Must be called before perfmon_init() but only if you want to use another
// access mode as the pre-configured one. For direct access (0) you have to
// be root.
//accessClient_setaccessmode(0);

// Initialize the perfmon module.
err = perfmon_init(topo->numHWThreads, cpus);
if (err < 0)
{
printf("Failed to initialize LIKWID's performance monitoring module\n");
topology_finalize();
return 1;
}

// Add eventset string to the perfmon module.
gid = perfmon_addEventSet(estr);
if (gid < 0)
{
printf("Failed to add event string %s to LIKWID's performance monitoring module\n", estr);
perfmon_finalize();
topology_finalize();
return 1;
}

// Setup the eventset identified by group ID (gid).
err = perfmon_setupCounters(gid);
if (err < 0)
{
printf("Failed to setup group %d in LIKWID's performance monitoring module\n", gid);
perfmon_finalize();
topology_finalize();
return 1;
}
// Start all counters in the previously set up event set.
err = perfmon_startCounters();
if (err < 0)
{
printf("Failed to start counters for group %d for thread %d\n",gid, (-1*err)-1);
perfmon_finalize();
topology_finalize();
return 1;
}

int r1= 100 , r2 = 1000;
int c1 = 1000 , c2 = 1000;

// Perform some work.


int A[r1][c1], B[r2][c2];

for(int i = 0 ; i<r1 ; i++){ 
for(int j = 0 ; j<c1 ; j++){

A[i][j] = i*c1 + j +1;
}
}
for(int i = 0 ; i<r2 ; i++){ 
for(int j = 0 ; j<c2 ; j++){

B[i][j] = i*c2 + j + 1;
}
}

int C[r1][c2];
for (int i = 0; i < r1; i++) {
for (int j = 0; j < c2; j++) {
C[i][j] = 0;

for (int k = 0; k < r2; k++) {
C[i][j] += A[i][k] * B[k][j];
}
}
}
printf("\nMatrix Mulitplication without threads\n");

// Read and record current event counts.
err = perfmon_readCounters();
if (err < 0)
{
printf("Failed to read counters for group %d for thread %d\n",gid, (-1*err)-1);
perfmon_finalize();
topology_finalize();
return 1;
}

// Print the result of every thread/CPU for all events in estr, counting from last read/startCounters().
printf("Work task 1/2 measurements:\n");
for (j=0; j<n; j++)
{
for (i = 0;i < topo->numHWThreads; i++)
{
result = perfmon_getLastResult(gid, j, i);
printf("- event set %s at CPU %d: %f\n", enames[j], cpus[i], result);
}
}


// Perform another piece of work
int nthreads = 5;
omp_set_num_threads(nthreads);
#pragma omp parallel
{	

int id = omp_get_thread_num();

for(int i = id ; i<r1*c2 ; i+=nthreads){

int row = i/c2;
int col = i%c2;

C[row][col] = 0;

for (int k = 0; k < r2; k++) {
C[row][col] += A[row][k] * B[k][col];
}
}    
}	

printf("\nMatrix Mulitplication with threads\n");

// Read and record current event counts.
err = perfmon_readCounters();
if (err < 0)
{
printf("Failed to read counters for group %d for thread %d\n",gid, (-1*err)-1);
perfmon_finalize();
topology_finalize();
return 1;
}

// Print the result of every thread/CPU for all events in estr, counting between the 
// previous two calls of perfmon_readCounters().
printf("Work task 2/2 measurements:\n");
for (j=0; j<n; j++)
{
for (i = 0;i < topo->numHWThreads; i++)
{
result = perfmon_getLastResult(gid, j, i);
printf("- event set %s at CPU %d: %f\n", enames[j], cpus[i], result);
}
}



// Stop all counters in the currently-active event set.
err = perfmon_stopCounters();
if (err < 0)
{
printf("Failed to stop counters for group %d for thread %d\n",gid, (-1*err)-1);
perfmon_finalize();
topology_finalize();
return 1;
}

// Print the result of every thread/CPU for all events in estr, counting since counters first started.
printf("Total sum measurements:\n");
for (j=0; j<n; j++)
{
for (i = 0;i < topo->numHWThreads; i++)
{
result = perfmon_getResult(gid, j, i);
printf("- event set %s at CPU %d: %f\n", enames[j], cpus[i], result);
}
}


free(cpus);
// Uninitialize the perfmon module.
perfmon_finalize();
affinity_finalize();
// Uninitialize the topology module.
topology_finalize();
return 0;
}
