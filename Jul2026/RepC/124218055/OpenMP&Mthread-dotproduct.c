/*****************************************************************************
* FILE: openmp_dotproduct.c
* DESCRIPTION:
*   This simple program is the a simple implementation version of OpenMP how 
*   to do a dot product of two matrix and to compare to a serial program 
* SOURCE: Hao LIU
* DATE: 22/04/2018
******************************************************************************/

/*****************************************************************************
* COMPLIE LINE:
*   gcc OpenMP&Mthread-dotproduct.c -lmthread -L./mthread/lib -pthread -Wall
* OBJECT OMP LINE:
*   #pragma omp parallel private(i,tid,psum) num_threads(threads)
*   #pragma omp for reduction(+:sum) 
******************************************************************************/
#include "mthread.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


/* Define length of dot product vectors and number of OpenMP threads */
#define VECLEN 100000000
#define sizeof_var( var ) ((size_t)(&(var)+1)-(size_t)(&(var)))
#define getName(var)  #var 


//Global environmental variables declaration
typedef enum{STATIC,DYNAMIC,RUNTIME} status_t;
status_t status;
double *a, *b;
int chunk;

/* A linked list void node prepared for firstprivate attributes */
struct Node
{
// Any data type can be stored in this node
// size actually contains the type information int is 4 double is 8
void *data;
size_t size;
struct Node *next;
};

/* Function to add a node at the beginning of Linked List.
This function expects a pointer to the data to be added
and size of the data type */
void push(struct Node** head_ref, void *new_data,char *s, size_t data_size)
{
// Allocate memory for node
struct Node* new_node = (struct Node*)malloc(sizeof(struct Node));

new_node->data = malloc(data_size);
new_node->next = (*head_ref);
new_node->size = data_size;

// Copy contents of new_data to newly allocated memory.
// Assumption: char takes 1 byte.
int i;
for (i=0; i<data_size; i++)
*(char *)(new_node->data + i) = *(char *)(new_data + i);

// Change head pointer as new node is added at the beginning
(*head_ref) = new_node;
}

/* Function to print nodes in a given linked list. fpitr is used
to access the function to be used for printing current node data.
Note that different data types need different specifier in printf() */
void printList(struct Node *node)
{
while (node != NULL)
{
if (node->size == 4)
printf(" %d", *(int *)node->data);
if (node->size == 8)
printf(" %lf", *(double *)node->data);
node = node->next;
}
}

void freeList(struct Node *node)
{
struct Node* tmp;
while (node != NULL)
{
tmp = node;
node = node->next;
free(tmp);
}
}
/* argument to mthread function except first_private void* list
* the rest of the part should be the private data to each thread
* since the shared variables is accessible for anyone
*/
struct arg_struct {
struct Node *first_private;
int tid;
int size;
int iter;
};

void schedule(status_t status_mode, int chunk_size){
status = status_mode;
chunk = chunk_size;
}

void *fun(void *args){
struct arg_struct *arg = (struct arg_struct *)(args);

int i,j,tid,psum;
tid = arg->tid;
//Check environment and decide schedule mode
if (status == STATIC){
for (i=tid,psum=0,j=0; j<arg->size;j++,i=i+chunk)
psum += a[i]*b[i];
}
printf("Thread %d partial sum = %d\n",tid, psum);
//printList(arg->first_private);
return (void*)psum;
}

int main(int argc, char** argv)
{   
//Get OMP_NUM_THREADS from system environment if not found set a default value 4
int NUM_THREADS;
const char* s = getenv("OMP_NUM_THREADS");
if (s != NULL)
NUM_THREADS = atoi(s);
else
NUM_THREADS = 4;
printf ("************************************************\n");
printf("#### OpenMP environment variables\n     OpenMP NUM threads = %d \n",NUM_THREADS);
/*Set OMP SCHEDULE environment and chunk size*/
schedule(STATIC, 1);
//print shedule mode and chunk size
switch(status){
case 0 :
printf("     Schedule mode = STATIC\n     Chunk size = %d\n", chunk);
break;
case 1 :
printf("     Schedule mode = DYNAMIC\n,     Chunk size = %d\n", chunk);
break;
case 2 :
printf("     Schedule mode = RUNTIME\n,      Chunk size = %d\n", chunk);
break;
default : 
break;
}
printf ("************************************************\n");

//private element
int i = 1;
int tid =12;
double psum=123.45;
// Create and print an void* linked list which can tackle any data type
printf("Firstprivate elements are %d, %d, %lf\n",i,tid,psum);
struct Node *first_private = NULL;
push(&first_private, &i, getName(i), sizeof_var(i));
push(&first_private, &tid, getName(tid), sizeof_var(tid));
push(&first_private, &psum, getName(psum), sizeof_var(psum));

printf("Created private linked list is");
printList(first_private);
//printf("%s", getName(psum));
printf ("\n************************************************\n");
int len=VECLEN, threads=NUM_THREADS;

/* Assign storage for dot product vectors */
a = (double*) malloc (len*threads*sizeof(double));
b = (double*) malloc (len*threads*sizeof(double));

/* Initialize dot product vectors */
for (int i=0; i<len*threads; i++) {
a[i]=1.0;
b[i]=a[i];
}

// Create arg for every each thread
struct arg_struct arg[NUM_THREADS];

for (int i = 0; i < threads; ++i)
{
arg[i].first_private = first_private;
arg[i].size = len;;
arg[i].iter = len*NUM_THREADS;
arg[i].tid = i;
}

struct timeval start, end;
double delta_paral,delta_seq;
gettimeofday(&start, NULL);
// OpenMP parallel region by creating a certain number of mthreads
// Inside of fun, OpenMP for is implemented
printf("Starting OpenMP parallel region\n%d threads will be launched\n",threads);
printf ("************************************************\n");
printf("Starting OpenMP for region\n");
mthread_t pid[NUM_THREADS];
for(int i = 0; i < NUM_THREADS; ++i){
mthread_create(&pid[i], NULL, fun, (void*)&arg[i]);
}

double sum_paral = 0.0;
double sum_seq = 0.0;

// reduction get particial sum 
double tsum[NUM_THREADS];
void* res;
for(int i = 0; i < NUM_THREADS; ++i){
mthread_join(pid[i],&res);
tsum[i] = (int)res;
}
//Add particial sum together to perform reduction operation
for(int i = 0; i < NUM_THREADS; ++i){
sum_paral+=tsum[i];
}
gettimeofday(&end, NULL);
delta_paral = ((end.tv_sec  - start.tv_sec) * 1000000u + 
end.tv_usec - start.tv_usec) / 1.e6;

//compare to sequential result
printf ("************************************************\n");
printf("Reduction operation all threads have been joined\n");

printf ("************************************************\n");
printf ("A sequencial computing version is launched\n");

gettimeofday(&start, NULL);
for (int i=0; i<len*threads; i++) 
{
sum_seq += a[i]*b[i];
}
printf ("************************************************\n");
gettimeofday(&end, NULL);
delta_seq = ((end.tv_sec  - start.tv_sec) * 1000000u + 
end.tv_usec - start.tv_usec) / 1.e6;

printf ("#### Comparison of results \n");
printf ("OpenMP version: sum  =  %f \n", sum_paral);
printf ("Sequencial version: sum  =  %f \n", sum_seq);
printf ("************************************************\n");
printf ("#### Comparison of performance \n");
printf ("     OpenMP version time %lf \n", delta_paral);
printf ("     Sequencial version time %lf \n", delta_seq);
free (a);
free (b);
freeList(first_private);

return 0;
}  