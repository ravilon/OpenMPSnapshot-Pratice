#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/*
 * Generates a random matrix with `seed`.
 */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Allocate space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. Remember to set all fieds of the matrix struct.
 * `parent` should be set to NULL to indicate that this matrix is not a slice.
 * You should return -1 if either `rows` or `cols` or both have invalid values, or if any
 * call to allocate memory in this function fails. If you don't set python error messages here upon
 * failure, then remember to set it in numc.c.
 * Return 0 upon success and non-zero upon failure.
 */
 /*
 typedef struct matrix {
  int rows;       // number of rows
  int cols;       // number of columns
  double **data;  // each element is a pointer to a row of data
  int is_1d;      // Whether this matrix is a 1d matrix
                  // For 1D matrix, shape is (rows * cols)
  int ref_cnt;
  struct matrix *parent;
} matrix;
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    if (rows < 1 || cols < 1) {
        return -1;
    }
    *mat = malloc(sizeof(matrix));
    if (!*mat) {
        return -1;
    }
    (*mat)->rows = rows;
    (*mat)->cols = cols;
    (*mat)->ref_cnt = 1;
    (*mat)->parent = NULL;
    // if rows = 1 or cols = 1, then is_1d is 1, o.w. 0.
    (*mat)->is_1d = (rows == 1 || cols == 1) ? 1 : 0;
    (*mat)->data = (double **) malloc(rows * sizeof(double *));
    (*mat)->data2 = (double *) calloc(rows * cols, sizeof(double));
    if (!(*mat)->data || !(*mat)->data2) {
        free(mat);
        return -1;
    }
    for (int i = 0; i < (*mat)->rows; i++) {
        (*mat)->data[i] = (*mat)->data2 + i * cols;
    }
    return 0;
}

/*
 * Allocate space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * This is equivalent to setting the new matrix to be
 * from[row_offset:row_offset + rows, col_offset:col_offset + cols]
 * If you don't set python error messages here upon failure, then remember to set it in numc.c.
 * Return 0 upon success and non-zero upon failure.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int row_offset, int col_offset,
                        int rows, int cols) {

    *mat = malloc(sizeof(matrix));
    if (!*mat) {
        return -1;
    }
    (*mat)->rows = rows;
    (*mat)->cols = cols;
    // if rows = 1 or cols = 1, then is_1d is 1, o.w. 0.
    (*mat)->is_1d = (rows == 1 || cols == 1) ? 1 : 0;
    (*mat)->data = (double **) malloc(rows * sizeof(double *));
    if (!(*mat)->data) {
        free(*mat);
        return -1;
    }



    (*mat)->parent = from;
    // (*mat)->ref_cnt = from->ref_cnt + 1;
    (*mat)->ref_cnt = 1;
    from->ref_cnt = from->ref_cnt + 1;
    // CHANGED r <= ROWS to r < ROWS!!
    for (int r = row_offset; r < row_offset + rows; r++) {
        for (int c = col_offset; c < col_offset + cols; c++) {
            (*mat)->data[r - row_offset] = from->data[r] + col_offset;
        }
    }
    return 0;
}

/*
 * This function will be called automatically by Python when a numc matrix loses all of its
 * reference pointers.
 * You need to make sure that you only free `mat->data` if no other existing matrices are also
 * referring this data array.
 * See the spec for more information.
 */
void deallocate_matrix(matrix *mat) {
    if (!mat) {
        return;
    }
    if (mat->ref_cnt == 1 || mat->ref_cnt < 0) {
        if (mat->parent) {
            if (mat->parent->ref_cnt == -2) {
                free(mat->parent->data);
                free(mat->parent->data2);
                free(mat->parent);
                free(mat->data);
                // free(mat->data2);
                free(mat);
            } else if (mat->parent->ref_cnt < -2) {
                mat->parent->ref_cnt += 1;
                free(mat->data);
                free(mat);
            } else {
                mat->parent->ref_cnt -= 1;
                free(mat->data);
                free(mat);
            }
        } else if (mat->ref_cnt < -1) {
            return;
        } else if (mat->ref_cnt == -1) {
            free(mat->data);
            free(mat->data2);
            free(mat);
        } else if (mat->ref_cnt == 1) {
            free(mat->data);
            free(mat->data2);
            free(mat);
        }
    } else if (mat->ref_cnt != 1) {
        if (mat->ref_cnt > 0) {
            mat->ref_cnt = mat->ref_cnt * -1;
        }
    }
}

/*
 * Return the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
    return mat->data[row][col];
}

/*
 * Set the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    mat->data[row][col] = val;
}

/*
 * Set all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {
    for (int r = 0; r < mat->rows; r++) {
        for(int c = 0; c < mat->cols; c++){
          mat->data[r][c] = val;
        }
    }
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    int cols = mat1->cols;
    #pragma omp parallel for
    for (int r = 0; r < mat1->rows; r++) {
        __m256d result1 = _mm256_setzero_pd();
        __m256d result2 = _mm256_setzero_pd();
        __m256d result3 = _mm256_setzero_pd();
        __m256d result4 = _mm256_setzero_pd();
        __m256d result5 = _mm256_setzero_pd();
        __m256d result6 = _mm256_setzero_pd();
        for(int c = 0; c < cols/24 * 24; c+=24){
            double *temp1 = mat1->data[r] + c;
            double *temp2 = mat2->data[r] + c;
            // m1
            __m256d m1rc1 = _mm256_loadu_pd(temp1);
			__m256d m1rc2 = _mm256_loadu_pd(temp1 + 4);
			__m256d m1rc3 = _mm256_loadu_pd(temp1 + 8);
			__m256d m1rc4 = _mm256_loadu_pd(temp1 + 12);
            __m256d m1rc5 = _mm256_loadu_pd(temp1 + 16);
			__m256d m1rc6 = _mm256_loadu_pd(temp1 + 20);
            // m2
            __m256d m2rc1 = _mm256_loadu_pd(temp2);
			__m256d m2rc2 = _mm256_loadu_pd(temp2 + 4);
			__m256d m2rc3 = _mm256_loadu_pd(temp2 + 8);
			__m256d m2rc4 = _mm256_loadu_pd(temp2 + 12);
            __m256d m2rc5 = _mm256_loadu_pd(temp2 + 16);
			__m256d m2rc6 = _mm256_loadu_pd(temp2 + 20);
            // result adding
            result1 = _mm256_sub_pd(m1rc1, m2rc1);
            result2 = _mm256_sub_pd(m1rc2, m2rc2);
            result3 = _mm256_sub_pd(m1rc3, m2rc3);
            result4 = _mm256_sub_pd(m1rc4, m2rc4);
            result5 = _mm256_sub_pd(m1rc5, m2rc5);
            result6 = _mm256_sub_pd(m1rc6, m2rc6);
            //#pragma omp critical {
            _mm256_storeu_pd(result->data[r] + c, result1);
            _mm256_storeu_pd(result->data[r] + c + 4, result2);
            _mm256_storeu_pd(result->data[r] + c + 8, result3);
            _mm256_storeu_pd(result->data[r] + c + 12, result4);
            _mm256_storeu_pd(result->data[r] + c + 16, result5);
            _mm256_storeu_pd(result->data[r] + c + 20, result6);
            //}
        }
        for (int i = cols/24 * 24; i < cols; i++) {
            result->data[r][i] = mat1->data[r][i] - mat2->data[r][i];
        }
    }
    return 0;
}


double **transpose(int rows, int cols, matrix *mat2) {
    double* mat2t = (double *) malloc(rows * cols * sizeof(double));
    double** mat2tp = (double **) malloc(cols * sizeof(double *));
    for (int x = 0; x < cols; x++) {
        mat2tp[x] = mat2t + x * rows;
    }
    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
            mat2tp[c][r] = mat2->data[r][c];
        }
    }
    return mat2tp;
}


/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    if (mat1->cols * mat1->rows > 10000) {
        double* dst = (double*) malloc(mat2->rows * mat2->cols * sizeof(double));
        int n = mat2->cols;
        int jump1 = 20;
        #pragma omp parallel for
        for(int r = 0; r < mat2->rows; r+= jump1) {
            for(int c = 0; c < n; c+= jump1) {
                for(int r2 = r; r2 < jump1 + r; r2++) {
                    for(int c2 = c; c2 < jump1 + c; c2++) {
                        if (r2 >= mat2->rows || c2 >= n) {
                            continue;
                        } else {
                            dst[r2 + c2 * mat2->rows] = mat2->data2[c2 + r2 * n];
                        }
                    }
                    if (r2 >= mat2->rows) {
                        continue;
                    }
                }
            }
        }

        #pragma omp parallel for
        for (int r = 0; r < mat1->rows; r++) {
            for (int c = 0; c < mat2->cols; c++) {
                // double *temp3 = result->data[r] + c;
                __m256d result1 = _mm256_setzero_pd();
                int flag = 0;
                for (int i = 0; i < mat1->cols/24 * 24; i+=24) {
                    flag = 1;
                    double *temp1 = mat1->data[r] + i;
                    double *temp2 = dst + i + c * mat2->rows;
                    __m256d m1rc1 = _mm256_loadu_pd(temp1);
                    __m256d m1rc2 = _mm256_loadu_pd(temp1 + 4);
                    __m256d m1rc3 = _mm256_loadu_pd(temp1 + 8);
                    __m256d m1rc4 = _mm256_loadu_pd(temp1 + 12);
                    __m256d m1rc5 = _mm256_loadu_pd(temp1 + 16);
                    __m256d m1rc6 = _mm256_loadu_pd(temp1 + 20);

                    __m256d m2rc1 = _mm256_loadu_pd(temp2);
                    __m256d m2rc2 = _mm256_loadu_pd(temp2 + 4);
                    __m256d m2rc3 = _mm256_loadu_pd(temp2 + 8);
                    __m256d m2rc4 = _mm256_loadu_pd(temp2 + 12);
                    __m256d m2rc5 = _mm256_loadu_pd(temp2 + 16);
                    __m256d m2rc6 = _mm256_loadu_pd(temp2 + 20);

                    result1 = _mm256_fmadd_pd(m1rc1, m2rc1, result1);
                    result1 = _mm256_fmadd_pd(m1rc2, m2rc2, result1);
                    result1 = _mm256_fmadd_pd(m1rc3, m2rc3, result1);
                    result1 = _mm256_fmadd_pd(m1rc4, m2rc4, result1);
                    result1 = _mm256_fmadd_pd(m1rc5, m2rc5, result1);
                    result1 = _mm256_fmadd_pd(m1rc6, m2rc6, result1);

                }
                if (flag == 1) {
                    double p[4];
                    _mm256_storeu_pd(p, result1);

                    result->data[r][c] = p[0] + p[1] + p[2] + p[3];
                    flag = 0;
                }

                for (int i = mat1->cols/24 * 24; i < mat1->cols; i++) {
                    result->data[r][c] = mat1->data[r][i] * dst[c * mat2->rows + i] + result->data[r][c];

                }
            }
        }
        free(dst);
    }
    else {
        for (int r = 0; r < mat1->rows; r++) {
            for (int i = 0; i < mat1->cols; i++) {
                for (int c = 0; c < mat2->cols; c++) {
                    result->data[r][c] = mat1->data[r][i] * mat2->data[i][c] + result->data[r][c];
                }
            }
        }
    }

    return 0;
}

int mul_matrix_pow(matrix *result, matrix *mat1, matrix *mat2) {

    #pragma omp parallel for
    for (int r = 0; r < mat1->rows; r++) {
        for(int i = 0; i < mat1->cols; i++) {
            for(int c = 0; c < mat2->cols; c++){
              //result->data[r][c] = mat1t[r][i] * mat2t[i][c] + result->data[r][c];
              result->data[r][c] = mat1->data[r][i] * mat2->data[i][c] + result->data[r][c];
              //result->data[r][c] = mat1data[r * mat1->cols + i] * mat2data[i * mat2->cols + c] + result->data[r][c];
            }
        }
    }
    return 0;

}


/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    if(pow <0){
        return -1;
    }else if (pow == 0) {
        for (int r = 0; r < mat->rows; r++) {
            for (int c = 0; c < mat->cols; c++) {
                if(c == r){
                    result->data[r][c] = 1;
                }else{
                    result->data[r][c] = 0;
                }
            }
        }
        return 0;
    }
    else if(pow == 1) {
        #pragma omp parallel for
        for (int r = 0; r < mat->rows; r++) {
            for (int c = 0; c < mat->cols; c++) {
                result->data[r][c] = mat->data[r][c];
            }
        }
    } else if(pow >= 2){


    matrix *temp_m = NULL;
    int alloc_failed = allocate_matrix(&temp_m, mat->rows, mat->cols);
    if (alloc_failed) {
        return -1;
    }
    matrix *temp_1 = NULL;
    alloc_failed = allocate_matrix(&temp_1, mat->rows, mat->cols);
    if (alloc_failed) {
        return -1;
    }
    matrix *temp_2 = NULL;
    alloc_failed = allocate_matrix(&temp_2, mat->rows, mat->cols);
    if (alloc_failed) {
        return -1;
    }

    #pragma omp parallel for
    for (int r = 0; r < mat->rows; r++) {
        for (int c = 0; c < mat->cols; c++) {
            temp_m->data[r][c] = mat->data[r][c];
            if(c == r){
                result->data[r][c] = 1;
            }else{
                result->data[r][c] = 0;
            }
        }
    }
    while(pow > 0){
      if (pow & 1) {
          #pragma omp parallel for
          for(int r = 0; r < mat->rows; r++){
              for(int c = 0; c < mat->cols; c++){
                  temp_1->data[r][c] = result->data[r][c];
                  result->data[r][c] = 0;
              }
          }
          mul_matrix_pow(result, temp_1, temp_m);
          if(pow == 1){
              break;
          }
      }
      pow = pow >> 1;
      #pragma omp parallel for
      for(int r = 0; r < mat->rows; r++){
          for(int c = 0; c < mat->cols; c++){
              temp_1->data[r][c] = temp_m->data[r][c];
              temp_2->data[r][c] = temp_m->data[r][c];
              temp_m->data[r][c] = 0;
          }
      }
      mul_matrix_pow(temp_m, temp_1, temp_2);
    }
    deallocate_matrix(temp_m);
    deallocate_matrix(temp_1);
    deallocate_matrix(temp_2);
  }
    return 0;
}

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    int cols = mat->cols;
    #pragma omp parallel for
    for (int r = 0; r < mat->rows; r++) {

        for(int c = 0; c < cols/24 * 24; c+=24){
            __m256d result1 = _mm256_set1_pd(-1);
            __m256d result2 = _mm256_set1_pd(-1);
            __m256d result3 = _mm256_set1_pd(-1);
            __m256d result4 = _mm256_set1_pd(-1);
            __m256d result5 = _mm256_set1_pd(-1);
            __m256d result6 = _mm256_set1_pd(-1);
            double *temp1 = mat->data[r] + c;
            // m1
            __m256d m1rc1 = _mm256_loadu_pd(temp1);
			__m256d m1rc2 = _mm256_loadu_pd(temp1 + 4);
			__m256d m1rc3 = _mm256_loadu_pd(temp1 + 8);
			__m256d m1rc4 = _mm256_loadu_pd(temp1 + 12);
            __m256d m1rc5 = _mm256_loadu_pd(temp1 + 16);
			__m256d m1rc6 = _mm256_loadu_pd(temp1 + 20);
            // result adding
            result1 = _mm256_mul_pd(m1rc1, result1);
            result2 = _mm256_mul_pd(m1rc2, result2);
            result3 = _mm256_mul_pd(m1rc3, result3);
            result4 = _mm256_mul_pd(m1rc4, result4);
            result5 = _mm256_mul_pd(m1rc5, result5);
            result6 = _mm256_mul_pd(m1rc6, result6);
            //#pragma omp critical {
            _mm256_storeu_pd(result->data[r] + c, result1);
            _mm256_storeu_pd(result->data[r] + c + 4, result2);
            _mm256_storeu_pd(result->data[r] + c + 8, result3);
            _mm256_storeu_pd(result->data[r] + c + 12, result4);
            _mm256_storeu_pd(result->data[r] + c + 16, result5);
            _mm256_storeu_pd(result->data[r] + c + 20, result6);
            //}
        }
        for (int i = cols/24 * 24; i < cols; i++) {
            result->data[r][i] = -1 * mat->data[r][i];
        }
    }
    return 0;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) {
    int cols = mat->cols;
    #pragma omp parallel for
    for (int r = 0; r < mat->rows; r++) {
        for(int c = 0; c < cols/24 * 24; c+=24){
            double *temp1 = mat->data[r] + c;
            __m256d result1 = _mm256_set1_pd(-1);
            __m256d result2 = _mm256_set1_pd(-1);
            __m256d result3 = _mm256_set1_pd(-1);
            __m256d result4 = _mm256_set1_pd(-1);
            __m256d result5 = _mm256_set1_pd(-1);
            __m256d result6 = _mm256_set1_pd(-1);
            // m1
            __m256d m1rc1 = _mm256_loadu_pd(temp1);
			__m256d m1rc2 = _mm256_loadu_pd(temp1 + 4);
			__m256d m1rc3 = _mm256_loadu_pd(temp1 + 8);
			__m256d m1rc4 = _mm256_loadu_pd(temp1 + 12);
            __m256d m1rc5 = _mm256_loadu_pd(temp1 + 16);
			__m256d m1rc6 = _mm256_loadu_pd(temp1 + 20);
            // result adding
            result1 = _mm256_mul_pd(m1rc1, result1);
            result2 = _mm256_mul_pd(m1rc2, result2);
            result3 = _mm256_mul_pd(m1rc3, result3);
            result4 = _mm256_mul_pd(m1rc4, result4);
            result5 = _mm256_mul_pd(m1rc5, result5);
            result6 = _mm256_mul_pd(m1rc6, result6);
            // result max
            result1 = _mm256_max_pd(m1rc1, result1);
            result2 = _mm256_max_pd(m1rc2, result2);
            result3 = _mm256_max_pd(m1rc3, result3);
            result4 = _mm256_max_pd(m1rc4, result4);
            result5 = _mm256_max_pd(m1rc5, result5);
            result6 = _mm256_max_pd(m1rc6, result6);
            _mm256_storeu_pd(result->data[r] + c, result1);
            _mm256_storeu_pd(result->data[r] + c + 4, result2);
            _mm256_storeu_pd(result->data[r] + c + 8, result3);
            _mm256_storeu_pd(result->data[r] + c + 12, result4);
            _mm256_storeu_pd(result->data[r] + c + 16, result5);
            _mm256_storeu_pd(result->data[r] + c + 20, result6);

        }
        for (int i = cols/24 * 24; i < cols; i++) {
            if(mat->data[r][i] < 0){
                result->data[r][i] = -1 * mat->data[r][i];
            }else{
                result->data[r][i] = mat->data[r][i];
            }
        }
    }
    return 0;
}
