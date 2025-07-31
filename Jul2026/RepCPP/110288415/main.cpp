#include <iostream>
#include <cmath>
#include "omp.h"

using namespace std;

typedef long long dlong;

int MAX_THREAD_NUMBER = omp_get_max_threads();
dlong MAX_VECTOR_SIZE = 15 * (dlong) 1e8;

dlong MAX_MATRIX_SIZE = 5 * (dlong) 1e4;

dlong MATRIX_SIZES[] = {(dlong) 1e3, dlong(1e4), MAX_MATRIX_SIZE};

void serial_scalar_multiplication(int *v1, int *v2, dlong size, dlong &result) {
result = 0;

for (dlong i = 0; i < size; i++) {
result += v1[i] * v2[i];
}
}

void parallel_scalar_multiplication(int *v1, int *v2, dlong size, int thread_number, dlong &result) {

result = 0;

#pragma omp parallel num_threads(thread_number)
#pragma omp for schedule(static) reduction(+:result)
for (dlong i = 0; i < size; i++) {
result += v1[i] * v2[i];
}
}

void init_vector(int *vect, dlong size) {
for (dlong i = 0; i < size; i++) {
vect[i] = -10 + (double) rand() * 20 / RAND_MAX;
}
}

void vector_task_run() {
int *v1 = new int[MAX_VECTOR_SIZE];
int *v2 = new int[MAX_VECTOR_SIZE];

init_vector(v1, MAX_VECTOR_SIZE);
init_vector(v2, MAX_VECTOR_SIZE);

dlong vector_sizes[] = {(dlong) 10e4, (dlong) 10e5, (dlong) 10e6, (dlong) 10e7,
(dlong) 10e8, MAX_VECTOR_SIZE};

double start, finish;
dlong result;

for (dlong &vector_size: vector_sizes) {
cout << "Vector size : " << vector_size << endl;

start = omp_get_wtime();

serial_scalar_multiplication(v1, v2, vector_size, result);

finish = omp_get_wtime();

cout << "Serial scalar time: " << finish - start << endl;

for (int thread_number = 8; thread_number <= MAX_THREAD_NUMBER; thread_number++) {
start = omp_get_wtime();

parallel_scalar_multiplication(v1, v2, vector_size, thread_number, result);

finish = omp_get_wtime();

cout << thread_number << " threads' time: " << finish - start << endl;
}
cout << endl;
}

delete[] v1;
delete[] v2;
}

void init_matrix(int **matrix, dlong size) {
for (dlong i = 0; i < size; i++) {
for (dlong j = 0; j < size; j++) {
matrix[i][j] = -10 + (double) rand() * 20 / RAND_MAX;
}
}
}

void print_main_results(double diag_average, dlong diag_min, dlong diag_max, dlong matrix_min, dlong matrix_max,
double time, int thread_number) {
cout << thread_number << " threads time: " << time << endl;
//cout << "Diagonal: average = " << diag_average << " , max = " << diag_max << " , min = " << diag_min << endl;
//cout << "Matrix:   max = " << matrix_min << " , min = " << matrix_min << endl;
}

void serial_process_matrix(int **matrix, dlong size) {
int matrix_max = matrix[0][0];
int matrix_min = matrix[0][0];

int *row_min = new int[size];
int *row_max = new int[size];
double *row_average = new double[size];

int diag_max = matrix[0][0];
int diag_min = matrix[0][0];
double diag_average = 0;
double start, finish;

start = omp_get_wtime();

for (dlong i = 0; i < size; i++) {

row_min[i] = matrix[i][0];
row_max[i] = matrix[i][0];
row_average[i] = 0;

for (dlong j = 0; j < size; j++) {
row_average[i] += matrix[i][j];

row_max[i] = max(row_max[i], matrix[i][j]);
row_min[i] = min(row_min[i], matrix[i][j]);

if (i == j) {
diag_average += matrix[i][j];
diag_max = max(diag_max, matrix[i][j]);
diag_min = min(diag_min, matrix[i][j]);
}
}
row_average[i] /= size;
matrix_max = max(matrix_max, row_max[i]);
matrix_min = min(matrix_min, row_min[i]);
}

diag_average /= size;
finish = omp_get_wtime();

print_main_results(diag_average, diag_min, diag_max, matrix_min, matrix_max, finish - start, 1);
}

void parallel_process_matrix(int **matrix, dlong size, int thread_number) {
int matrix_max = matrix[0][0];
int matrix_min = matrix[0][0];

int *row_min = new int[size];
int *row_max = new int[size];
double *row_average = new double[size];

int diag_max = matrix[0][0];
int diag_min = matrix[0][0];
double diag_average = 0;
double start, finish;

start = omp_get_wtime();

#pragma omp parallel num_threads(thread_number)
#pragma omp for schedule(static) reduction(+:diag_average)

for (dlong i = 0; i < size; i++) {

row_min[i] = matrix[i][0];
row_max[i] = matrix[i][0];
row_average[i] = 0;

for (dlong j = 0; j < size; j++) {
row_average[i] += matrix[i][j];

row_max[i] = max(row_max[i], matrix[i][j]);
row_min[i] = min(row_min[i], matrix[i][j]);

if (i == j) {
diag_average += matrix[i][j];
diag_max = max(diag_max, matrix[i][j]);
diag_min = min(diag_min, matrix[i][j]);
}
}
row_average[i] /= size;
matrix_max = max(matrix_max, row_max[i]);
matrix_min = min(matrix_min, row_min[i]);
}

diag_average /= size;
finish = omp_get_wtime();

print_main_results(diag_average, diag_min, diag_max, matrix_min, matrix_max, finish - start, thread_number);

}

void destroy_matrix(int **matrix, dlong size) {
for (dlong i = 0; i < size; i++) {
delete[] matrix[i];
}
delete[] matrix;
}

void matrix_task_run() {
int **matrix;
matrix = new int *[MAX_MATRIX_SIZE];
for (int i = 0; i < MAX_MATRIX_SIZE; i++) matrix[i] = new int[MAX_MATRIX_SIZE];

init_matrix(matrix, MAX_MATRIX_SIZE);

for (dlong &matrix_size: MATRIX_SIZES) {
cout << "Matrix size: " << matrix_size << " x " << matrix_size << endl;
serial_process_matrix(matrix, matrix_size);
for (int thread_num = 1; thread_num <= MAX_THREAD_NUMBER; thread_num++) {
parallel_process_matrix(matrix, matrix_size, thread_num);
}
cout << endl;
}

destroy_matrix(matrix, MAX_MATRIX_SIZE);
}


int main() {
srand((unsigned int) time(NULL));
cout.precision(10);
vector_task_run();

return 0;
}