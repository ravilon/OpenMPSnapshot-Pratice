#define MATRICES_1D_OPEN_MP
#ifdef MATRICES_1D_OPEN_MP

/*                  FINAL VERSION
* Functions don't allocate arrays that they return. */

/* Matrices are represented as 1-D arrays in memory.
* That means they are contiguous in memory, flat arrays.
* Minimum dimension is 1, not 0, and internal dimensions must match. */

/* All functions use restricted pointers, so care should be taken
* to make sure that arrays that they point to do not overlap, if
* we want to modify them inside of the functions.
* On the other hand, it's easy to change type of the pointers
* from restricted to non-restricted versions, by using definitions
* given at the beginning of the corresponding "matrices_1d.h"
* header file, if necessary. */

/* Uses tiles to speed up computations, by using cache efficiently.
* This makes sense when working with matrices; in particular, with
* operations that traverse columns, like dot () and transpose(). */

#include "matrices_1d.h"
#include "tests.h"


/* Initializes vector or matrix with sequentially growing data_t values, starting from 0. */
void init_seq(data_ptr_res_t a, const unsigned n_rows_a, const unsigned n_cols_a) {
int i = 0, j = 0;

#pragma omp parallel for default(none) private(i, j) shared(a, n_rows_a, n_cols_a) schedule(static)
for (i = 0; i < (int)n_rows_a; i++) {
for (j = 0; j < (int)n_cols_a; j++) {
a[i*n_cols_a + j] = i*n_cols_a + j;
}    
}
}

/* Initializes vector or matrix with sequentially growing data_t values, starting from 0. */
void init_seq_tiled(data_ptr_res_t a, const unsigned n_rows_a, const unsigned n_cols_a) {
int i = 0, j = 0, it = 0, jt = 0;

#pragma omp parallel for default(none) private(i, j, it, jt) shared(a, n_rows_a, n_cols_a) schedule(static)
for (i = 0; i < (int)n_rows_a; i += TILE_ORDER) {
for (j = 0; j < (int)n_cols_a; j += TILE_ORDER) {
for (it = i; it < MIN((int)n_rows_a, i + TILE_ORDER); it++) {
for (jt = j; jt < MIN((int)n_cols_a, j + TILE_ORDER); jt++) {
a[it*n_cols_a + jt] = it*n_cols_a + jt;
}
}
}
}
}

/* Initializes vector or matrix, with random data_t values in the range [0, 1].
Lot slower than init_seq(), which is expected, since it calls rand(). */
void init_rand(data_ptr_res_t a, const unsigned n_rows_a, const unsigned n_cols_a) {
int i = 0, j = 0;

/* Schedule should be either guided or dynamic; if it's static or runtime, the random numbers may repeat.
But, if working with tiles, it might not work with guided or dynamic, but also should have
less problems with repeating values. */
#pragma omp parallel for default(none) private(i, j) shared(a, n_rows_a, n_cols_a) schedule(static)
for (i = 0; i < (int)n_rows_a; i++) {
for (j = 0; j < (int)n_cols_a; j++) {
a[i*n_cols_a + j] = rand() / (data_t)RAND_MAX;
}
}     
}

/* Initializes vector or matrix, with random data_t values in the range [0, 1].
Lot slower than init_seq(), which is expected, since it calls rand(). */
void init_rand_tiled(data_ptr_res_t a, const unsigned n_rows_a, const unsigned n_cols_a) {
int i = 0, j = 0, it = 0, jt = 0;

/* Schedule should be either guided or dynamic; if it's static or runtime, the random numbers may repeat.
But, if working with tiles, it might not work with guided or dynamic, but also should have
less problems with repeating values. */
#pragma omp parallel for default(none) private(i, j, it, jt) shared(a, n_rows_a, n_cols_a) schedule(static)
for (i = 0; i < (int)n_rows_a; i += TILE_ORDER) {
for (j = 0; j < (int)n_cols_a; j += TILE_ORDER) {
for (it = i; it < MIN((int)n_rows_a, i + TILE_ORDER); it++) {
for (jt = j; jt < MIN((int)n_cols_a, j + TILE_ORDER); jt++) {
a[it*n_cols_a + jt] = rand() / (data_t)RAND_MAX;
}
}
}
}
}

/* Sum of an array */
data_t sum_array(cdata_ptr_res_t arr, const unsigned length) {
data_t sum = 0.;
int i = 0;

#pragma omp parallel for default(none) private (i) shared(arr, length) reduction(+:sum) schedule(static)
for (i = 0; i < (int)length; i++) {
sum += arr[i];
}

return sum;
}

/* Sum of an array */
data_t sum_array_tiled(cdata_ptr_res_t arr, const unsigned length) {
data_t sum = 0.;
int i = 0, it = 0;

#pragma omp parallel for default(none) private (i, it) shared(arr, length) reduction(+:sum) schedule(static)
for (i = 0; i < (int)length; i += TILE_ORDER) {
for (it = i; it < MIN((int)length, i + TILE_ORDER); it++) {
sum += arr[it];
}
}

return sum;
}

/* Mean value of an array */
data_t mean(cdata_ptr_res_t arr, const unsigned length) {
data_t sum = 0.;
int i = 0;

#pragma omp parallel for default(none) private (i) shared(arr, length) reduction(+:sum) schedule(static)
for (i = 0; i < (int)length; i++) {
sum += arr[i];
}

return sum / length;
}

/*  Takes and returns a new matrix, t, which is a transpose of the original one, m.
It's also flat in memory, i.e., 1-D, but it should be looked at as a transpose
of m, meaning, n_rows_t == n_cols_m, and n_cols_t == n_rows_m.
The original matrix m stays intact. */
data_ptr_res_t transpose(cdata_ptr_res_t m, const unsigned n_rows_m, const unsigned n_cols_m, data_ptr_res_t t) {
int i = 0, j = 0, it = 0, jt = 0;

#pragma omp parallel for default(none) private(i, j, it, jt) shared(m, n_rows_m, n_cols_m, t) schedule(static)
for (i = 0; i < (int)n_rows_m; i += TILE_ORDER) {
for (j = 0; j < (int)n_cols_m; j += TILE_ORDER) {
for (it = i; it < MIN((int)n_rows_m, i + TILE_ORDER); it++) {
for (jt = j; jt < MIN((int)n_cols_m, j + TILE_ORDER); jt++) {
t[jt*n_rows_m + it] = m[it*n_cols_m + jt];
}
}
}
}

return t;
}

/*  Takes and returns a new matrix, t, which is a transpose of the original one, m.
It's also flat in memory, i.e., 1-D, but it should be looked at as a transpose
of m, meaning, n_rows_t == n_cols_m, and n_cols_t == n_rows_m.
The original matrix m stays intact. */
data_ptr_res_t transpose_non_tiled(cdata_ptr_res_t m, const unsigned n_rows_m, const unsigned n_cols_m, data_ptr_res_t t) {
int i = 0, j = 0;

#pragma omp parallel for default(none) private(i, j) shared(m, n_rows_m, n_cols_m, t) schedule(static)
for (i = 0; i < (int)n_rows_m; i++) {
for (j = 0; j < (int)n_cols_m; j++) {
t[j*n_rows_m + i] = m[i*n_cols_m + j];
}
}

/* Visual validation - Prints t like m, the original */
const int validate = 0;
if (validate) {
for (size_t i = 0; i < n_rows_m; i++) {
for (size_t j = 0; j < n_cols_m; j++) {
printf("%8.3f ", t[j*n_rows_m + i]);
}
printf("\n");
}
printf("\n");
}

return t;
}

/* Dot product of two arrays, a and b, or matrix product
* Returns an array that's passed in as the last argument, c.
* This is by far the slowest version of the function, sequentially or parallely. */
data_ptr_res_t dot_simple(cdata_ptr_res_t a, const unsigned n_rows_a, const unsigned n_cols_a, \
cdata_ptr_res_t b, const unsigned n_rows_b, const unsigned n_cols_b, data_ptr_res_t c) {

/* Check lengths of the input arrays */
if (n_cols_a != n_rows_b) {
printf("#columns A must be equal to #rows B!\n");
system("pause");
exit(-2);
}

int i = 0, j = 0, k = 0;

#pragma omp parallel for default(none) private(i, j, k) shared(a, n_rows_a, n_cols_a, b, n_rows_b, n_cols_b, c) schedule(static)
for (i = 0; i < (int)n_rows_a; i++) {
for (k = 0; k < (int)n_cols_b; k++) {
data_t sum = 0.0;
for (j = 0; j < (int)n_cols_a; j++) {
sum += a[i*n_cols_a + j] * b[j*n_cols_b + k];
}
c[i*n_cols_b + k] = sum;
}
}

return c;
}

/* Dot product of two arrays, a and b, or matrix product
* Returns an array that's passed in as the last argument, c.
* This is a tiled version of the simple function, and it's much faster than it. */
data_ptr_res_t dot_simple_tiled(cdata_ptr_res_t a, const unsigned n_rows_a, const unsigned n_cols_a, \
cdata_ptr_res_t b, const unsigned n_rows_b, const unsigned n_cols_b, data_ptr_res_t c) {

/* Check lengths of the input arrays */
if (n_cols_a != n_rows_b) {
printf("#columns A must be equal to #rows B!\n");
system("pause");
exit(-2);
}

int i = 0, j = 0, k = 0, it = 0, jt = 0, kt = 0;

memset(c, 0, n_rows_a * n_cols_b * sizeof(*c));

#pragma omp parallel for default(none) private(i, j, k, it, jt, kt) shared(a, n_rows_a, n_cols_a, b, n_rows_b, n_cols_b, c) schedule(static)
for (i = 0; i < (int)n_rows_a; i += TILE_ORDER) {
for (k = 0; k < (int)n_cols_b; k += TILE_ORDER) {
for (j = 0; j < (int)n_cols_a; j += TILE_ORDER) {
for (it = i; it < MIN((int)n_rows_a, i + TILE_ORDER); it++) {
for (kt = k; kt < MIN((int)n_cols_b, k + TILE_ORDER); kt++) {
data_t sum = 0.0;
for (jt = j; jt < MIN((int)n_cols_a, j + TILE_ORDER); jt++) {
sum += a[it*n_cols_a + jt] * b[jt*n_cols_b + kt];
}
c[it*n_cols_b + kt] += sum;
}
}
}
}
}

return c;
}

/* Dot product of two arrays, a and b, or matrix product
* Returns an array that's passed in as the last argument, c.
* This is a much faster version of the function.
* Uses more memory than the simple version, for transposing matrix b, which
* can be a problem if the matrix is large - there might not be enough memory.
* It's the fastest one, sequential or Open MP, if we optimize for speed. */
data_ptr_res_t dot_faster(cdata_ptr_res_t a, const unsigned n_rows_a, const unsigned n_cols_a, \
cdata_ptr_res_t b, const unsigned n_rows_b, const unsigned n_cols_b, data_ptr_res_t c) {

/* Check lengths of the input arrays */
if (n_cols_a != n_rows_b) {
printf("#columns A must be equal to #rows B!\n");
system("pause");
exit(-2);
}

int i = 0, j = 0, k = 0;

data_ptr_res_t bt = malloc(n_rows_b * n_cols_b * sizeof(*b));

if (!bt) {
printf("Couldn't allocate memory!\n");
system("pause");
exit(-1);
}

bt = transpose(b, n_rows_b, n_cols_b, bt);

#pragma omp parallel for default(none) private(i, j, k) shared(a, n_rows_a, n_cols_a, b, n_rows_b, n_cols_b, c, bt) schedule(static)
for (i = 0; i < (int)n_rows_a; i++) {
for (k = 0; k < (int)n_cols_b; k++) {
data_t sum = 0.0;
for (j = 0; j < (int)n_cols_a; j++) {
sum += a[i*n_cols_a + j] * bt[k*n_rows_b + j];
}
c[i*n_cols_b + k] = sum;
}
}

free(bt);

return c;
}

/* Dot product of two arrays, a and b, or matrix product
* Returns an array that's passed in as the last argument, c.
* Uses more memory than the simple version, for transposing matrix b, which
* can be a problem if the matrix is large - there might not be enough memory.
* This was supposed to be the fastest version of the function,
* but it's similar in speed to dot_simple_tiled, if we optimize for speed.
* But, if we optimize for the smallest code, this version is the fastest,
* though, not faster than dot_faster optimized for speed, but of the same speed as it. */
data_ptr_res_t dot_faster_tiled(cdata_ptr_res_t a, const unsigned n_rows_a, const unsigned n_cols_a, \
cdata_ptr_res_t b, const unsigned n_rows_b, const unsigned n_cols_b, data_ptr_res_t c) {

/* Check lengths of the input arrays */
if (n_cols_a != n_rows_b) {
printf("#columns A must be equal to #rows B!\n");
system("pause");
exit(-2);
}

int i = 0, j = 0, k = 0, it = 0, jt = 0, kt = 0;

data_ptr_res_t bt = malloc(n_rows_b * n_cols_b * sizeof(*b));

if (!bt) {
printf("Couldn't allocate memory!\n");
system("pause");
exit(-1);
}

bt = transpose(b, n_rows_b, n_cols_b, bt);

memset(c, 0, n_rows_a * n_cols_b * sizeof(*c));

#pragma omp parallel for default(none) private(i, j, k, it, jt, kt) shared(a, n_rows_a, n_cols_a, b, n_rows_b, n_cols_b, c, bt) schedule(static)
for (i = 0; i < (int)n_rows_a; i += TILE_ORDER) {
for (k = 0; k < (int)n_cols_b; k += TILE_ORDER) {
for (j = 0; j < (int)n_cols_a; j += TILE_ORDER) {
for (it = i; it < MIN((int)n_rows_a, i + TILE_ORDER); it++) {
for (kt = k; kt < MIN((int)n_cols_b, k + TILE_ORDER); kt++) {
data_t sum = 0.0;
for (jt = j; jt < MIN((int)n_cols_a, j + TILE_ORDER); jt++) {
sum += a[it*n_cols_a + jt] * bt[kt*n_rows_b + jt];
}
c[it*n_cols_b + kt] += sum;
}
}
}
}
}

free(bt);

return c;
}

/*  Adds two arrays, element-wise, and puts the result in an array
that is passed in as the last argument, and also returns it.
Arrays must be of the same length, or, one of them, or both, can be scalars.
Use 0 as the length of a scalar, and pass its address in (a pointer to it). */
data_ptr_res_t add_arrays(cdata_ptr_res_t a, const unsigned n_a, cdata_ptr_res_t b, const unsigned n_b, data_ptr_res_t result) {
/* Check lengths of the input arrays */
if ((n_a != n_b) && (n_a != 0) && (n_b != 0)) {
printf("Length of A must be equal to length of B!\n");
system("pause");
exit(-2);
}

int i = 0;

/* Neither a nor b are scalars. */
if ((n_a > 0) && (n_b > 0)) {        
#pragma omp parallel for default(none) private(i) shared(a, n_a, b, n_b, result) schedule(static)
for (i = 0; i < (int)n_a; i++) {
result[i] = a[i] + b[i];
}
}
/* Only b is scalar. */
else if ((n_b == 0) && (n_a > 0)) {
#pragma omp parallel for default(none) private(i) shared(a, n_a, b, n_b, result) schedule(static)
for (i = 0; i < (int)n_a; i++) {
result[i] = a[i] + *b;
}
}
/* Only a is scalar. */
else if ((n_a == 0) && (n_b > 0)) {
#pragma omp parallel for default(none) private(i) shared(a, n_a, b, n_b, result) schedule(static)
for (i = 0; i < (int)n_b; i++) {
result[i] = *a + b[i];
}
}
/* Both a and b are scalars. */
else {
result[0] = *a + *b;
}

return result;
}

/*  Subtracts the second array from the first one, element-wise, and puts the result
in an array that is passed in as the last argument, and also returns it.
Arrays must be of the same length, or, one of them, or both, can be scalars.
Use 0 as the length of a scalar, and pass its address in (a pointer to it). */
data_ptr_res_t subtract_arrays(cdata_ptr_res_t a, const unsigned n_a, cdata_ptr_res_t b, const unsigned n_b, data_ptr_res_t result) {
/* Check lengths of the input arrays */
if ((n_a != n_b) && (n_a != 0) && (n_b != 0)) {
printf("Length of A must be equal to length of B!\n");
system("pause");
exit(-2);
}

int i = 0;

/* Neither a nor b are scalars. */
if ((n_a > 0) && (n_b > 0)) {
#pragma omp parallel for default(none) private(i) shared(a, n_a, b, n_b, result) schedule(static)
for (i = 0; i < (int)n_a; i++) {
result[i] = a[i] - b[i];
}
}
/* Only b is scalar. */
else if ((n_b == 0) && (n_a > 0)) {
#pragma omp parallel for default(none) private(i) shared(a, n_a, b, n_b, result) schedule(static)
for (i = 0; i < (int)n_a; i++) {
result[i] = a[i] - *b;
}
}
/* Only a is scalar. */
else if ((n_a == 0) && (n_b > 0)) {
#pragma omp parallel for default(none) private(i) shared(a, n_a, b, n_b, result) schedule(static)
for (i = 0; i < (int)n_b; i++) {
result[i] = *a - b[i];
}
}
/* Both a and b are scalars. */
else {
result[0] = *a - *b;
}

return result;
}

/*  Multiplies two arrays, element-wise, and puts the result in an array
that is passed in as the last argument, and also returns it.
Arrays must be of the same length, or, one of them, or both, can be scalars.
Use 0 as the length of a scalar, and pass its address in (a pointer to it).
Tiled version is slightly slower in Open MP, and evidently slower sequentially. */
data_ptr_res_t multiply_arrays(cdata_ptr_res_t a, const unsigned n_a, cdata_ptr_res_t b, const unsigned n_b, data_ptr_res_t result) {
/* Check lengths of the input arrays */
if ((n_a != n_b) && (n_a != 0) && (n_b != 0)) {
printf("Length of A must be equal to length of B!\n");
system("pause");
exit(-2);
}

int i = 0;

/* Neither a nor b are scalars. */
if ((n_a > 0) && (n_b > 0)) {
#pragma omp parallel for default(none) private(i) shared(a, n_a, b, n_b, result) schedule(static)
for (i = 0; i < (int)n_a; i++) {
result[i] = a[i] * b[i];
}
}
/* Only b is scalar. */
else if ((n_b == 0) && (n_a > 0)) {
#pragma omp parallel for default(none) private(i) shared(a, n_a, b, n_b, result) schedule(static)
for (i = 0; i < (int)n_a; i++) {
result[i] = a[i] * *b;
}
}
/* Only a is scalar. */
else if ((n_a == 0) && (n_b > 0)) {
#pragma omp parallel for default(none) private(i) shared(a, n_a, b, n_b, result) schedule(static)
for (i = 0; i < (int)n_b; i++) {
result[i] = *a * b[i];
}
}
/* Both a and b are scalars. */
else {
result[0] = *a * *b;
}

return result;
}

/*  Multiplies two arrays, element-wise, and puts the result in an array
that is passed in as the last argument, and also returns it.
Arrays must be of the same length, or, one of them, or both, can be scalars.
Use 0 as the length of a scalar, and pass its address in (a pointer to it).
Tiled version is slightly slower in Open MP, and evidently slower sequentially. */
data_ptr_res_t multiply_arrays_tiled(cdata_ptr_res_t a, const unsigned n_a, cdata_ptr_res_t b, const unsigned n_b, data_ptr_res_t result) {
/* Check lengths of the input arrays */
if ((n_a != n_b) && (n_a != 0) && (n_b != 0)) {
printf("Length of A must be equal to length of B!\n");
system("pause");
exit(-2);
}

int i = 0, it = 0;

/* Neither a nor b are scalars. */
if ((n_a > 0) && (n_b > 0)) {
#pragma omp parallel for default(none) private(i, it) shared(a, n_a, b, n_b, result) schedule(static)
for (i = 0; i < (int)n_a; i += TILE_ORDER) {
for (it = i; it < MIN((int)n_a, i + TILE_ORDER); it++) {
result[it] = a[it] * b[it];
}
}
}
/* Only b is scalar. */
else if ((n_b == 0) && (n_a > 0)) {
#pragma omp parallel for default(none) private(i, it) shared(a, n_a, b, n_b, result) schedule(static)
for (i = 0; i < (int)n_a; i += TILE_ORDER) {
for (it = i; it < MIN((int)n_a, i + TILE_ORDER); it++) {
result[it] = a[it] * *b;
}
}
}
/* Only a is scalar. */
else if ((n_a == 0) && (n_b > 0)) {
#pragma omp parallel for default(none) private(i, it) shared(a, n_a, b, n_b, result) schedule(static)
for (i = 0; i < (int)n_b; i += TILE_ORDER) {
for (it = i; it < MIN((int)n_b, i + TILE_ORDER); it++) {
result[it] = *a * b[it];
}
}
}
/* Both a and b are scalars. */
else {
result[0] = *a * *b;
}

return result;
}

/*  Divides two arrays, element-wise, and puts the result in an array
that is passed in as the last argument, and also returns it.
Arrays must be of the same length, or, one of them, or both, can be scalars.
Use 0 as the length of a scalar, and pass its address in (a pointer to it). */
data_ptr_res_t divide_arrays(cdata_ptr_res_t a, const unsigned n_a, cdata_ptr_res_t b, const unsigned n_b, data_ptr_res_t result) {
/* Check lengths of the input arrays */
if ((n_a != n_b) && (n_a != 0) && (n_b != 0)) {
printf("Length of A must be equal to length of B!\n");
system("pause");
exit(-2);
}

int i = 0;

/* Neither a nor b are scalars. */
if ((n_a > 0) && (n_b > 0)) {
#pragma omp parallel for default(none) private(i) shared(a, n_a, b, n_b, result) schedule(static)
for (i = 0; i < (int)n_a; i++) {
result[i] = a[i] / b[i];
}
}
/* Only b is scalar. */
else if ((n_b == 0) && (n_a > 0)) {
#pragma omp parallel for default(none) private(i) shared(a, n_a, b, n_b, result) schedule(static)
for (i = 0; i < (int)n_a; i++) {
result[i] = a[i] / *b;
}
}
/* Only a is scalar. */
else if ((n_a == 0) && (n_b > 0)) {
#pragma omp parallel for default(none) private(i) shared(a, n_a, b, n_b, result) schedule(static)
for (i = 0; i < (int)n_b; i++) {
result[i] = *a / b[i];
}
}
/* Both a and b are scalars. */
else {
result[0] = *a / *b;
}

return result;
}

/*  Updates an array, element-wise, by adding another array to it.
Takes both arrays in, and returns the updated one (the first one).
The return value (address of the first array) doesn't have to be used.
Arrays must be of the same length, or, the second one can be a scalar.
Use 0 as the length of a scalar, and pass its address in (a pointer to it). */
data_ptr_res_t add_update(data_ptr_res_t a, const unsigned n_a, cdata_ptr_res_t b, const unsigned n_b) {
/* Check lengths of the input arrays */
if (n_a == 0) {
printf("'A' cannot be a scalar!\n");
system("pause");
exit(-2);
}
if ((n_a != n_b) && (n_b != 0)) {
printf("Length of A must be equal to length of B!\n");
system("pause");
exit(-2);
}

int i = 0;

/* b is scalar */
if (n_b == 0) {
#pragma omp parallel for default(none) private(i) shared(a, n_a, b, n_b) schedule(static)
for (i = 0; i < (int)n_a; i++) {
a[i] += *b;
}
}
/* b is array */
else {
#pragma omp parallel for default(none) private(i) shared(a, n_a, b, n_b) schedule(static)
for (i = 0; i < (int)n_a; i++) {
a[i] += b[i];
}
}

return a;
}

/*  Compares two arrays element-wise, and puts the result in an array
that is passed in as the last argument, and also returns it.
If an element of array a is greater than a corresponding element of
array b, the resulting array will have 1.0 in that position;
it will have 0.0 otherwise.
Arrays must be of the same length, or, one of them, or both, can be scalars.
Use 0 as the length of a scalar, and pass its address in (a pointer to it). */
data_ptr_res_t greater_than(cdata_ptr_res_t a, const unsigned n_a, cdata_ptr_res_t b, const unsigned n_b, data_ptr_res_t result) {
/* Check lengths of the input arrays */
if ((n_a != n_b) && (n_a != 0) && (n_b != 0)) {
printf("Length of A must be equal to length of B!\n");
system("pause");
exit(-2);
}

int i = 0;

/* Neither a nor b are scalars. */
if ((n_a > 0) && (n_b > 0)) {
#pragma omp parallel for default(none) private(i) shared(a, n_a, b, n_b, result) schedule(static)
for (i = 0; i < (int)n_a; i++) {
result[i] = a[i] > b[i];
}
}
/* Only b is scalar. */
else if ((n_b == 0) && (n_a > 0)) {
#pragma omp parallel for default(none) private(i) shared(a, n_a, b, n_b, result) schedule(static)
for (i = 0; i < (int)n_a; i++) {
result[i] = a[i] > *b;
}
}
/* Only a is scalar. */
else if ((n_a == 0) && (n_b > 0)) {
#pragma omp parallel for default(none) private(i) shared(a, n_a, b, n_b, result) schedule(static)
for (i = 0; i < (int)n_b; i++) {
result[i] = *a > b[i];
}
}
/* Both a and b are scalars. */
else {
result[0] = *a > *b;
}

return result;
}

/*  Compares two arrays element-wise, and puts the result in an array
that is passed in as the last argument, and also returns it.
If an element of array a is equal to a corresponding element of
array b, the resulting array will have 1.0 in that position;
it will have 0.0 otherwise.
Arrays must be of the same length, or, one of them, or both, can be scalars.
Use 0 as the length of a scalar, and pass its address in (a pointer to it). */
data_ptr_res_t equal(cdata_ptr_res_t a, const unsigned n_a, cdata_ptr_res_t b, const unsigned n_b, data_ptr_res_t result) {
/* Check lengths of the input arrays */
if ((n_a != n_b) && (n_a != 0) && (n_b != 0)) {
printf("Length of A must be equal to length of B!\n");
system("pause");
exit(-2);
}

int i = 0;

/* Neither a nor b are scalars. */
if ((n_a > 0) && (n_b > 0)) {
#pragma omp parallel for default(none) private(i) shared(a, n_a, b, n_b, result) schedule(static)
for (i = 0; i < (int)n_a; i++) {
result[i] = a[i] == b[i];
}
}
/* Only b is scalar. */
else if ((n_b == 0) && (n_a > 0)) {
#pragma omp parallel for default(none) private(i) shared(a, n_a, b, n_b, result) schedule(static)
for (i = 0; i < (int)n_a; i++) {
result[i] = a[i] == *b;
}
}
/* Only a is scalar. */
else if ((n_a == 0) && (n_b > 0)) {
#pragma omp parallel for default(none) private(i) shared(a, n_a, b, n_b, result) schedule(static)
for (i = 0; i < (int)n_b; i++) {
result[i] = *a == b[i];
}
}
/* Both a and b are scalars. */
else {
result[0] = *a == *b;
}

return result;
}

/* Prints vector, or matrix. */
void print(cdata_ptr_res_t m, const unsigned n_rows_m, const unsigned n_cols_m) {
for (size_t i = 0; i < n_rows_m; i++) {
for (size_t j = 0; j < n_cols_m; j++) {
printf("%8.3f ", m[i*n_cols_m + j]);
}
printf("\n");
}
printf("\n");
}

/* Sequential function for comparing two arrays by using memcmp
Returns 0 if contents of the arrays are the same; -1 or 1 otherwise. */
int compare_memcmp(cdata_ptr_res_t a, const unsigned n_a, cdata_ptr_res_t b, const unsigned n_b) {
/* Check lengths of the input arrays */
if (n_a != n_b) {
printf("Length of A must be equal to length of B!\n");
system("pause");
exit(-2);
}

return memcmp(a, b, n_a);
}

/* Sequential function for comparing two arrays by using a loop
Returns 0 if contents of the arrays are the same; 1 otherwise. */
int compare(cdata_ptr_res_t a, const unsigned n_a, cdata_ptr_res_t b, const unsigned n_b) {
/* Check lengths of the input arrays */
if (n_a != n_b) {
printf("Length of A must be equal to length of B!\n");
system("pause");
exit(-2);
}

for (size_t i = 0; i < n_a; i++) {
if (fabs(a[i] - b[i]) > TOLERANCE) {
return 1;
}
}

return 0;
}

/* Compares two scalars within a given TOLERANCE
Returns 0 if contents of the arrays are the same; 1 otherwise. */
int compare_scalars(const data_t a, const data_t b) {
if (fabs(a - b) > TOLERANCE) {
return 1;
}
return 0;
}


int main(int argc, char *argv[]) {
/* Intializes random number generator */
time_t t;
srand((unsigned)time(&t));
srand(0);

omp_set_num_threads(8);

printf("\tOPEN MP IMPLEMENTATION\n\n");

printf("omp_get_num_procs %i\n", omp_get_num_procs());
printf("omp_get_max_threads %i\n", omp_get_max_threads());
#pragma omp parallel
#pragma omp single
printf("Working with %d threads.\n\n", omp_get_num_threads());

test();

system("pause");
return(0);
}

#endif // MATRICES_1D_OPEN_MP
