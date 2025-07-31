/***************************
* File name: eight_queen.c
* Purpose: Solve the eight queen problem, and practice multi-thread programming
* Author: fdmdkw
* History:
*     * recursive version release: 2019/04/03
*     * multi-thread version release: 2019/04/30
*     * openmp version release: 2019/05/27
****************************/
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

#define PRINT_ANS 0 // print ans or not
void eq_recur(int arr[]);
void eq_recur_gettime(void);
void *eq_pthread(void *eq);
void eq_pthread_gettime(void);
void eq_omp(int arr[]);
void eq_omp_gettime(void);

int Q;
int counter = 0;
int thread_num = 0;
pthread_t *threads;
pthread_mutex_t count_lock = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
int *arr;
int thread_id;
}eq_thread_t;

int main(int argc, char *argv[])
{
if(argc != 3){
perror("Must input your number of queen and number of thread\n");
exit(EXIT_FAILURE);
}
// Q queen
Q = atoi(argv[1]);
if(Q < 4){
perror("Queen number must at least 4\n");
exit(EXIT_FAILURE);
}
// thread number
thread_num = atoi(argv[2]);
if(thread_num < 1){
perror("Thread number must at least 1\n");
exit(EXIT_FAILURE);
}

//eq_recur_gettime();
//eq_pthread_gettime();
eq_omp_gettime();

return 0;
}

void eq_recur(int arr[])
{
int now = arr[0];
for(int i = 1; i <= Q; i++){
int conflict = 0;
// check previos queen position
// previos point (x, y) = (j, arr[j])
// now point (x, y) = (now, i)
for(int j = 1; j < now; j++){
if(j + arr[j] == now + i || j - arr[j] == now - i || arr[j] == i){
conflict = 1;
break;
}
}
if(conflict)
continue;
arr[now] = i;
if(now == Q){
++counter;
if(PRINT_ANS){
printf("Ans %d: ", counter);
for(int k = 1; k <= Q; k++)
printf("%d ", arr[k]);
printf("\n");
}
return;
}
else{
int *a = malloc(sizeof(int) * (Q+1));
for(int j = 0; j <= Q; j++)
a[j] = arr[j];
++a[0]; // next point
eq_recur(a);
}
}
free(arr);
}

void eq_recur_gettime(void)
{
// timer
double start_utime,end_utime;
struct timeval tv, tv2;
// single thread of recursive version
// counter reset
counter = 0;
// int array
int *arr = malloc(sizeof(int)*(Q+1));
// initialize arr
for(int i = 1; i <= Q; i++)
arr[i] = 0;
arr[0] = 1;
// setup timer
gettimeofday(&tv,NULL);  
start_utime = tv.tv_sec * 1000000 + tv.tv_usec;
//sigle process without  multithread
eq_recur(arr);
// end timer
gettimeofday(&tv2,NULL); 
end_utime = tv2.tv_sec * 1000000 + tv2.tv_usec;
// output
puts("Sigle Process");
printf("Queen %d has %d number of answers\n", Q, counter);
printf("Sigle Process Execution Time = %f(s)\n", (end_utime - start_utime)/1000000);

// print a line
puts("---------------------------");
}

void *eq_pthread(void *eq)
{
int thread_id = ((eq_thread_t *)eq)->thread_id;
int *arr = ((eq_thread_t *)eq)->arr;

int now = arr[0];
for(int i = 1; i <= Q; i++){
if((i % thread_num) != thread_id && now == 1)
continue;
int conflict = 0;
// check previos queen position
// previos point (x, y) = (j, arr[j])
// now point (x, y) = (now, i)
for(int j = 1; j < now; j++){
if(j + arr[j] == now + i || j - arr[j] == now - i || arr[j] == i){
conflict = 1;
break;
}
}
if(conflict)
continue;
arr[now] = i;
if(now == Q){
pthread_mutex_lock(&count_lock);
++counter;
if(PRINT_ANS){
printf("Ans %d: ", counter);
for(int k = 1; k <= Q; k++)
printf("%d ", arr[k]);
printf("\n");
}
pthread_mutex_unlock(&count_lock);
//return;
}
else{
eq_thread_t tmp;
// tmp initialize
tmp.arr = (int*)malloc(sizeof(int)*(Q+1));
for(int j = 0; j <= Q; j++)
tmp.arr[j] = arr[j];
++tmp.arr[0];  // next point
tmp.thread_id = thread_id;

eq_pthread((void*)&tmp);
}
}
free(arr);
}

void eq_pthread_gettime(void)
{
// pthread version
// timer
double start_utime,end_utime;
struct timeval tv, tv2;
counter = 0;
// eq_thread_t array
eq_thread_t *eq = malloc(sizeof(eq_thread_t)*thread_num);

// eq initialize
for(int i = 0; i < thread_num; i++){
eq[i].arr = (int*)malloc(sizeof(int)*(Q+1));
for(int j = 1; j <= Q; j++)
eq[i].arr[j] = 0;
eq[i].arr[0] = 1;  // first element is now point
eq[i].thread_id = i;
}

// pthread_t array
threads = (pthread_t *)malloc(sizeof(pthread_t)*thread_num);

// setup timer
gettimeofday(&tv,NULL);  
start_utime = tv.tv_sec * 1000000 + tv.tv_usec;
// excute eq_pthread with multithreading
for(int i = 0; i < thread_num; i++)
pthread_create(&threads[i], NULL, eq_pthread, (void*)&eq[i]);
// join all threads
for(int i = 0; i < thread_num; i++)
pthread_join(threads[i], NULL);
// end timer
gettimeofday(&tv2,NULL);
end_utime = tv2.tv_sec * 1000000 + tv2.tv_usec;
// output
printf("Thread number: %d with pthread\n", thread_num);
printf("Queen %d has %d number of answers\n", Q, counter);
printf("Parallel Execution Time = %f(s)\n", (end_utime - start_utime)/1000000);

// print a line
puts("---------------------------");
}

void eq_omp(int arr[])
{
int now = arr[0];
for(int i = 1; i <= Q; i++){
int conflict = 0;
// check previos queen position
// previos point (x, y) = (j, arr[j])
// now point (x, y) = (now, i)
for(int j = 1; j < now; j++){
if(j + arr[j] == now + i || j - arr[j] == now - i || arr[j] == i){
conflict = 1;
break;
}
}
if(conflict)
continue;
arr[now] = i;
if(now == Q){
#pragma omp atomic
++counter;
if(PRINT_ANS){
printf("Ans %d: ", counter);
for(int k = 1; k <= Q; k++)
printf("%d ", arr[k]);
printf("\n");
}
return;
}
else{
int *a = malloc(sizeof(int) * (Q+1));
for(int j = 0; j <= Q; j++)
a[j] = arr[j];
++a[0]; // next point
eq_omp(a);
}
}
free(arr);
}

void eq_omp_gettime(void)
{
// timer
double start_utime,end_utime;
struct timeval tv, tv2;
// openmp version
// counter reset
counter = 0;

// setup timer
gettimeofday(&tv,NULL);  
start_utime = tv.tv_sec * 1000000 + tv.tv_usec;
// omp muti-thread
#pragma omp parallel for num_threads(thread_num) shared(counter)
for(int i = 1; i <= Q; i++){
// int array
int *arr = malloc(sizeof(int)*(Q+1));
// initialize omp_arr
for(int j = 2; j <= Q; j++)
arr[j] = 0;
arr[0] = 2;
arr[1] = i;
eq_omp(arr);
}
// end timer
gettimeofday(&tv2,NULL); 
end_utime = tv2.tv_sec * 1000000 + tv2.tv_usec;
// output
printf("Thread number: %d with openmp\n", thread_num);
printf("Queen %d has %d number of answers\n", Q, counter);
printf("Parallel Execution Time = %f(s)\n", (end_utime - start_utime)/1000000);

// print a line
puts("---------------------------");
}
