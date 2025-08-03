#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <arm_neon.h> /* NEON header */
#include <time.h>

/* Size Definitions */
#define SIZE_1 10000000
#define SIZE_2 20000000
#define SIZE_3 30000000

/* Vector definitions with float neon type */
float32_t y_1[SIZE_1], y_2[SIZE_2], y_3[SIZE_3];
float32_t x_1[SIZE_1], x_2[SIZE_2], x_3[SIZE_3];
float32x4_t a;


/* Main function */
int main ()
{
/* Time variables */
double start_time, run_time;
/* Time seed for random numbers */
srand(time(NULL));

omp_set_num_threads(omp_get_num_procs());

/* A Constant */
float32_t cnt = rand();
a = vdupq_n_f32(cnt);

/* Fill vector with random floating point numbers */ 
for (int i = 0; i < SIZE_3; ++i){
if(i < SIZE_1){
x_1[i] = rand();
y_1[i] = rand();
}
if (i < SIZE_2){
x_2[i] = rand();
y_2[i] = rand();
}
x_3[i] = rand();
y_3[i] = rand();
}

/* Time start with vector 1 */
start_time = omp_get_wtime();

/* OpenMP parallel section */
#pragma omp parallel  
{
/* Parallel for */
/* Compute SAXPY to vector 1 with neon extension*/
#pragma omp for
for (int i = 0; i < SIZE_1; i+=4){
float32x4_t y_vec = vld1q_f32(y_1+i); /* Load vector y */
float32x4_t x_vec = vld1q_f32(x_1+i); /* Load vector x */
y_vec = vmlaq_f32(y_vec, a, x_vec); /* Compute y + a*x */
vst1q_f32(y_1+i, y_vec); /* Store result in memory */
}
}
run_time = omp_get_wtime() - start_time;
printf("%d\t%f\n", SIZE_1, run_time);

start_time = omp_get_wtime();
/* OpenMP parallel section */
#pragma omp parallel  
{
/* Parallel for */
/* Compute SAXPY to vector 2 with neon extension*/
#pragma omp for
for (int i = 0; i < SIZE_2; i+=4){
float32x4_t y_vec = vld1q_f32(y_2+i); /* Load vector y */
float32x4_t x_vec = vld1q_f32(x_2+i); /* Load vector x */
y_vec = vmlaq_f32(y_vec, a, x_vec); /* Compute y + a*x */
vst1q_f32(y_2+i, y_vec); /* Store result in memory */
}
}
run_time = omp_get_wtime() - start_time;
printf("%d\t%f\n", SIZE_2, run_time);

start_time = omp_get_wtime();
/* OpenMP parallel section */
#pragma omp parallel  
{
/* Parallel for */
/* Compute SAXPY to vector 2 with neon extension*/
#pragma omp for
for (int i = 0; i < SIZE_3; i+=4){
float32x4_t y_vec = vld1q_f32(y_3+i); /* Load vector y */
float32x4_t x_vec = vld1q_f32(x_3+i); /* Load vector x */
y_vec = vmlaq_f32(y_vec, a, x_vec); /* Compute y + a*x */
vst1q_f32(y_3+i, y_vec); /* Store result in memory */
}
}
run_time = omp_get_wtime() - start_time;
printf("%d\t%f\n", SIZE_3, run_time);

return 0;
}