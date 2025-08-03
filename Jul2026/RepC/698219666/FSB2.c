/*
In order to facilitate testing, conditional compilation is used to produce various versions of FSB programs.
compile for no openmp no mpi:
gcc -o FSB2_sequential FSB2.c -lm
compile for openmp no mpi:
gcc -fopenmp -o FSB2_OpenMP FSB2.c -lm
compile for no openmp mpi:
mpicc -D USE_MPI -o FSB2_MPI FSB2.c -lm
compile for openmp mpi:
mpicc -D USE_MPI -fopenmp -o FSB2_OpenMP_MPI FSB2.c -lm
*/
#define DEFAULT_ITERATE_STEP (10LL)
#define DEFAULT_NUM_FISH (10000000LL)

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#if defined(USE_MPI)
#include <mpi.h>
#endif
#if defined(_OPENMP)
#include <omp.h>
#endif

#define INIT_WEIGHT (1.0)
#define MAX_WEIGHT (2 * INIT_WEIGHT)
#define MIN_XY (-100.0)
#define MAX_XY (100.0)
#define MIN_MOVING_XY (-0.1)
#define MAX_MOVING_XY (0.1)
#define RESULT_FILENAME "Testing Results/OMP_MPI_results.csv"

#if defined(USE_MPI)
#define INIT_TAG (888)
#define INIT_TAG_SELF_EXCHANGE (999)
#endif

typedef struct
{
double x;
double y;
} FISH;

typedef struct
{
double w;
double d;
double deltaf;
} FISHEXT;

static unsigned int global_seed;
#if defined(_OPENMP)
#pragma omp threadprivate(global_seed)
#endif

static double generate_range_random_double(double min, double max)
{
return ((((double)rand_r(&global_seed) / (double)RAND_MAX) * (max - min)) + min);
}

static double get_wtime()
{
struct timespec mtime;

clock_gettime(CLOCK_MONOTONIC, &mtime);

return ((double)mtime.tv_sec + ((double)mtime.tv_nsec / 1000000000.0));
}

#if defined(USE_MPI)
static void dump_to_file(const char *path, FISH *fish, long long int size)
{
FILE *F;
long long int i;

if ((path != NULL) && (fish != NULL))
{
F = fopen(path, "w+t");

if (F != NULL)
{
for (i = 0LL; i < size; i++)
{
fprintf(F, "%f %f\n", fish[i].x, fish[i].y);
}
fclose(F);
}
}
}
#endif

const char *get_schedule_state(char *str, size_t strsize)
{
if (str != NULL)
{
#if defined(_OPENMP)
omp_sched_t sched_kind;
int chunk_size;

omp_get_schedule(&sched_kind, &chunk_size);
switch (sched_kind & 0x7FFFFFFF)
{
case omp_sched_static:
snprintf(str, strsize, "static, %d", chunk_size);
break;
case omp_sched_dynamic:
snprintf(str, strsize, "dynamic, %d", chunk_size);
break;
case omp_sched_guided:
snprintf(str, strsize, "guided, %d", chunk_size);
break;
case omp_sched_auto:
snprintf(str, strsize, "auto, %d", chunk_size);
break;
default:
snprintf(str, strsize, "unknown(%x), %d", sched_kind, chunk_size);
break;
}
#else
snprintf(str, strsize, "n/a, n/a");
#endif
}

return str;
}

int main(int argc, char **argv)
{
FISH *fish;		  // Fish coordinate array pointer
FISHEXT *fishext; // Fish extend array pointer
int num_threads, rank, size, is_openmp, is_mpi;
long long int num_fishes, num_iterate_steps, i, t, m, m1, mx, b;
double sum_f;
double sum_f_i;
double max_delt_f;
double tmp_d;
double bari;
double t1, t2; // Time for recording
FILE *F;
char str[64];
#if defined(USE_MPI)
FISH *g_fish;	// All Fish coordinate array pointer
FISH *tempfish; // Only self for exchange
int j, max_items, curr_items;
long long int offset, curr_offset;
double mpi_reduction_sum_f, mpi_reduction_sum_f_i, mpi_reduction_max_delt_f, *p_mpi_scatter_max_delt_f;
#endif

#if defined(USE_MPI)
MPI_Init(&argc, &argv);

MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

if (size < 2)
{
printf("Need at least two MPI tasks. Quitting...\n");
exit(1);
}
is_mpi = 1;
#else
rank = 0;
size = 1;
is_mpi = 0;
#endif

#if !defined(_OPENMP)
is_openmp = 0;
#else
is_openmp = 1;
#endif

num_iterate_steps = DEFAULT_ITERATE_STEP;
num_fishes = DEFAULT_NUM_FISH;
if (argc > 1)
{
num_iterate_steps = atoll(argv[1]); // get steps from argument 1
if (num_iterate_steps < 0LL)
{
num_iterate_steps = DEFAULT_ITERATE_STEP;
}
if (argc > 2)
{
num_fishes = atoll(argv[2]); // get fish from argument 2
if (num_fishes <= 0LL)
{
num_fishes = DEFAULT_NUM_FISH;
}
#if defined(_OPENMP) // openmp model
if (argc > 3)
{
if (atoi(argv[3]) > 0LL)
{
omp_set_num_threads(atoi(argv[3]));
}
}
#endif
}
}

m = num_fishes / size;
b = num_fishes % size;
if (b != 0)
{
m1 = m + 1;
}
else
{
m1 = m;
}
mx = rank < b ? m1 : m;

fish = NULL;
fishext = NULL;
#if defined(USE_MPI)
g_fish = NULL;
tempfish = NULL;
p_mpi_scatter_max_delt_f = NULL;
#endif

fish = (FISH *)malloc(sizeof(FISH) * (size_t)mx); // heap alloc loacl fish array
if (fish == NULL)
{
printf("Error: failed to allocate buffer\n");
exit(1);
}

#if defined(USE_MPI)
if (rank == 0)
{
g_fish = (FISH *)malloc(sizeof(FISH) * (size_t)num_fishes); // heap alloc globel fish array
if (g_fish == NULL)
{
printf("Error: failed to allocate buffer\n");
exit(1);
}
#if defined(_OPENMP) // openmp model
#pragma omp parallel
{
global_seed = time(NULL) + rank + omp_get_thread_num();
#pragma omp for schedule(static)
#else
global_seed = time(NULL) + rank;
#endif
for (i = 0; i < num_fishes; i++)
{
g_fish[i].x = generate_range_random_double(MIN_XY, MAX_XY);
g_fish[i].y = generate_range_random_double(MIN_XY, MAX_XY);
}
#if defined(_OPENMP) // openmp model
}
#endif
printf("Dumping first file...\n");
dump_to_file(argc > 4 ? argv[4] : "fish1.txt", g_fish, num_fishes);
}

MPI_Barrier(MPI_COMM_WORLD);

max_items = INT_MAX / sizeof(FISH);
if ((m == m1) && (num_fishes <= max_items))
{
if (rank == 0)
{
printf("Exchange Data By Scatter/Gather Model...\n");
}
MPI_Scatter(g_fish, m * 2, MPI_DOUBLE, fish, m * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Gather(fish, m * 2, MPI_DOUBLE, g_fish, m * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}
else
{
if (rank == 0)
{
printf("Exchange Data By MPI_Send/MPI_Recv Model...\n");
}
max_items = INT_MAX / 2;
offset = 0LL;
do
{ // This loop is for large amounts of data
if (rank == 0)
{ // rank 0, using rank 1 for self exchange
curr_items = ((m1 - offset) > max_items ? max_items : (m1 - offset));
MPI_Send(g_fish + offset, curr_items * 2, MPI_DOUBLE, 1, INIT_TAG_SELF_EXCHANGE, MPI_COMM_WORLD);				   // send global fish block to rank 1
memset(g_fish + offset, 0, sizeof(FISH) * (size_t)curr_items);													   // clear sent global fish block
MPI_Recv(fish + offset, curr_items * 2, MPI_DOUBLE, 1, INIT_TAG_SELF_EXCHANGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // recv local fish block from rank 1
}
else if (rank == 1)
{ // rank 1
curr_items = ((m1 - offset) > max_items ? max_items : (m1 - offset));
if (tempfish == NULL)
{
tempfish = (FISH *)malloc(sizeof(FISH) * curr_items);
if (tempfish == NULL)
{
printf("Error: failed to allocate buffer\n");
exit(1);
}
}
MPI_Recv(tempfish, curr_items * 2, MPI_DOUBLE, 0, INIT_TAG_SELF_EXCHANGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // recv global fish block from rank 0
MPI_Send(tempfish, curr_items * 2, MPI_DOUBLE, 0, INIT_TAG_SELF_EXCHANGE, MPI_COMM_WORLD);					  // send global fish block to rank 0
}
if (rank == 0)
{
for (j = 1; j < size; j++)
{
if (j < b)
{
curr_items = (m1 - offset) > max_items ? max_items : (m1 - offset);
curr_offset = (m1 * j) + offset;
}
else
{
curr_items = (m - offset) > max_items ? max_items : (m - offset);
curr_offset = ((m1 * b) + (m * (j - b))) + offset;
}
MPI_Send(g_fish + curr_offset, curr_items * 2, MPI_DOUBLE, j, INIT_TAG, MPI_COMM_WORLD);					// send global fish block to rank j
memset(g_fish + curr_offset, 0, sizeof(FISH) * (size_t)curr_items);											// clear sent global fish block
MPI_Recv(g_fish + curr_offset, curr_items * 2, MPI_DOUBLE, j, INIT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // recv local fish block from rank j
}
}
else
{
if (rank < b)
{
curr_items = (m1 - offset) > max_items ? max_items : (m1 - offset);
}
else
{
curr_items = (m - offset) > max_items ? max_items : (m - offset);
}
MPI_Recv(fish + offset, curr_items * 2, MPI_DOUBLE, 0, INIT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // recv global fish block from rank 0
MPI_Send(fish + offset, curr_items * 2, MPI_DOUBLE, 0, INIT_TAG, MPI_COMM_WORLD);					 // send local fish block to rank 0
}
if (rank == 0)
{ // rank 0, using rank 1 for self exchange
curr_items = (m1 - offset) > max_items ? max_items : (m1 - offset);
MPI_Send(fish + offset, curr_items * 2, MPI_DOUBLE, 1, INIT_TAG_SELF_EXCHANGE, MPI_COMM_WORLD);						 // send local fish block to rank 1
MPI_Recv(g_fish + offset, curr_items * 2, MPI_DOUBLE, 1, INIT_TAG_SELF_EXCHANGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // recv global fish block from rank 1
}
else if (rank == 1)
{
curr_items = (m1 - offset) > max_items ? max_items : (m1 - offset);
MPI_Recv(tempfish, curr_items * 2, MPI_DOUBLE, 0, INIT_TAG_SELF_EXCHANGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // recv local fish block from rank 0
MPI_Send(tempfish, curr_items * 2, MPI_DOUBLE, 0, INIT_TAG_SELF_EXCHANGE, MPI_COMM_WORLD);					  // send local fish block to rank 0
}
offset += curr_items;
} while (offset < mx);
}

if (tempfish != NULL)
{
free(tempfish);
tempfish = NULL;
}

if (rank == 0)
{
printf("Dumping second file...\n");
dump_to_file(argc > 5 ? argv[5] : "fish2.txt", g_fish, num_fishes);
free(g_fish);
g_fish = NULL;

p_mpi_scatter_max_delt_f = (double *)malloc(sizeof(double) * size);
if (p_mpi_scatter_max_delt_f == NULL)
{
printf("Error: failed to allocate buffer\n");
exit(1);
}
}
#endif

fishext = (FISHEXT *)malloc(sizeof(FISHEXT) * (size_t)mx); // heap alloc loacl fish array
if (fishext == NULL)
{
printf("Error: failed to allocate buffer\n");
exit(1);
}

#if defined(_OPENMP) // openmp model
#pragma omp parallel
{
if (omp_get_thread_num() == 0)
{
num_threads = omp_get_num_threads();
}
#pragma omp for schedule(static)
#else
num_threads = 1;
#endif
for (i = 0; i < mx; i++)
{
#if !defined(USE_MPI)
fish[i].x = generate_range_random_double(MIN_XY, MAX_XY);
fish[i].y = generate_range_random_double(MIN_XY, MAX_XY);
#endif
fishext[i].w = INIT_WEIGHT;
fishext[i].d = sqrt((fish[i].x * fish[i].x) + (fish[i].y * fish[i].y));
}
#if defined(_OPENMP) // openmp model
}
#endif

#if defined(USE_MPI)
MPI_Barrier(MPI_COMM_WORLD);

mpi_reduction_sum_f_i = 1.0;
mpi_reduction_sum_f = 1.0;
#endif
bari = 0.0;
t1 = get_wtime();
for (t = 0; t < num_iterate_steps; t++)
{ // Every fish's moving coordinate depend on previous coordinate, so this loop can't parallel.
sum_f = 0.0;
sum_f_i = 0.0;
max_delt_f = -1.0;
if (t == 0)
{
#if defined(_OPENMP) // openmp model
#pragma omp parallel for schedule(runtime) reduction(+ : sum_f_i, sum_f)
#endif
for (i = 0; i < mx; i++)
{
sum_f += fishext[i].d;
fishext[i].w += generate_range_random_double(-0.0001, 0.0001); // first change weight
sum_f_i += fishext[i].d * fishext[i].w;
}
}
else
{
#if defined(_OPENMP) // openmp model
#pragma omp parallel
{
#pragma omp for schedule(runtime) private(tmp_d) reduction(max : max_delt_f)
#endif
for (i = 0; i < mx; i++)
{ // Move
do
{
fish[i].x += generate_range_random_double(MIN_MOVING_XY, MAX_MOVING_XY);
if (fish[i].x < MIN_XY)
{ // check for left limit
fish[i].x = MIN_XY;
}
else if (fish[i].x > MAX_XY)
{ // check for right limit
fish[i].x = MAX_XY;
}
fish[i].y += generate_range_random_double(MIN_MOVING_XY, MAX_MOVING_XY);
if (fish[i].y < MIN_XY)
{ // check for bottom limit
fish[i].y = MIN_XY;
}
else if (fish[i].y > MAX_XY)
{ // check for top limit
fish[i].y = MAX_XY;
}
tmp_d = sqrt((fish[i].x * fish[i].x) + (fish[i].y * fish[i].y));
fishext[i].deltaf = tmp_d - fishext[i].d;
} while (fishext[i].deltaf == 0.0); // max_delt_f from fish[i].deltaf, prevents division by zero
if (max_delt_f < fabs(fishext[i].deltaf))
{
max_delt_f = fabs(fishext[i].deltaf);
}
fishext[i].d = tmp_d;
}
#if defined(_OPENMP) // openmp model
}
#endif
#if defined(USE_MPI)
MPI_Reduce(&max_delt_f, &mpi_reduction_max_delt_f, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
if (rank == 0)
{
for (j = 0; j < size; j++)
{
p_mpi_scatter_max_delt_f[j] = mpi_reduction_max_delt_f;
}
}
MPI_Scatter(p_mpi_scatter_max_delt_f, 1, MPI_DOUBLE, &max_delt_f, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
#if defined(_OPENMP) // openmp model
#pragma omp parallel
{
#pragma omp for schedule(runtime) reduction(+ : sum_f_i, sum_f)
#endif
for (i = 0; i < mx; i++)
{ // Eat
sum_f += fishext[i].d;
fishext[i].w += (fishext[i].deltaf / max_delt_f);
if (fishext[i].w < 0.0)
{
fishext[i].w = 0.0;
}
else if (fishext[i].w > MAX_WEIGHT)
{ // check for over weight
fishext[i].w = MAX_WEIGHT;
}
sum_f_i += fishext[i].d * fishext[i].w;
}
#if defined(_OPENMP) // openmp model
}
#endif
}
#if defined(USE_MPI)
MPI_Reduce(&sum_f, &mpi_reduction_sum_f, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
MPI_Reduce(&sum_f_i, &mpi_reduction_sum_f_i, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
#if defined(USE_MPI)
if (rank == 0)
#if defined(_OpenMP)
if (omp_get_thread_num() == 0)
#endif
{
bari = mpi_reduction_sum_f_i / mpi_reduction_sum_f; // Get the result
}
#else
#if defined(_OpenMP)
if (omp_get_thread_num() == 0)
#endif
{
bari = sum_f_i / sum_f; // Get the result
}
#endif
}
t2 = get_wtime();
#if defined(USE_MPI)
if (rank == 0)
#elif defined(_OpenMP)
if (omp_get_thread_num() == 0)
#endif
{
printf("Barycenter: %f, Elapsed time(s): %f, Number of processes: %d, Number of threads: %d, Iterate times: %lld, Number of fishes: %lld\n", bari, t2 - t1, size, num_threads, num_iterate_steps, num_fishes);
F = fopen(RESULT_FILENAME, "at");
if (F != NULL)
{
fprintf(F, "%s, %s, %s, %f, %f, %d, %d, %lld, %lld\n", is_mpi ? "yes" : "no", is_openmp ? "yes" : "no", get_schedule_state(str, sizeof(str)), bari, t2 - t1, size, num_threads, num_iterate_steps, num_fishes);
fclose(F);
}
}

if (fish != NULL)
{
free(fish);
fish = NULL;
}

if (fishext != NULL)
{
free(fishext);
fishext = NULL;
}

#if defined(USE_MPI)
if (p_mpi_scatter_max_delt_f != NULL)
{
free(p_mpi_scatter_max_delt_f);
p_mpi_scatter_max_delt_f = NULL;
}

MPI_Finalize();
#endif

return 0;
}
