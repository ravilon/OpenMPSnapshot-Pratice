/**
 * matrixOp.c - performs various matrix opertations
 *
 * Written by Nicholas Cannon (22241579)
 */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "parseMatrix.h"

/**
 * Peforms a scalar multiplication on matrix mat and stores the answer in matrix
 * ans. Matrix ans must be the exact same as mat e.g. same rows, cols and
 * nzitems.
 *
 * Returns non zero on failure.
 */
int scalarMultiplication(COO *mat, COO *ans, float scalar) {
    int i;
    *ans = *mat;  // copy over meta info for answer matrix

    if (mat->type == MAT_FLOAT) {
#pragma omp parallel for
        for (i = 0; i < mat->nzsize; i++) {
            ((COO_ENTRY_FLOAT *)ans->NZ[i])->val *= scalar;
        }
    } else {
#pragma omp parallel for
        for (i = 0; i < mat->nzsize; i++) {
            ((COO_ENTRY_INT *)ans->NZ[i])->val *= scalar;
        }
    }

    return 0;
}

/**
 * Performs a trace on an INTEGER matrix and stores the answer in the ans
 * variable. Trace is the sum of each diagonal element. Note: this function is
 * only defined for square matrices.
 *
 * Returns non-zero on failure.
 */
int trace_INT(COO *mat, int *ans) {
    // basic input sanitization
    if (mat->type != MAT_INT || mat->cols != mat->rows) {
        return -1;
    }

    int trace_sum = 0;
    int i;
#pragma omp parallel for reduction(+ : trace_sum)
    for (i = 0; i < mat->nzsize; i++) {
        if (mat->NZ[i]->col == mat->NZ[i]->row) {
            trace_sum += ((COO_ENTRY_INT *)mat->NZ[i])->val;
        }
    }

    *ans = trace_sum;
    return 0;
}

/**
 * Performs a trace on an FLOAT matrix and stores the answer in the ans
 * variable. Trace is the sum of each diagonal element. Note: this function is
 * only defined for square matrices.
 *
 * Returns non-zero on failure.
 */
int trace_FLOAT(COO *mat, float *ans) {
    // basic input sanitization
    if (mat->type != MAT_FLOAT || mat->cols != mat->rows) {
        return -1;
    }

    float trace_sum = 0.0;
    int i;
#pragma omp parallel for reduction(+ : trace_sum)
    for (i = 0; i < mat->nzsize; i++) {
        if (mat->NZ[i]->col == mat->NZ[i]->row) {
            trace_sum += ((COO_ENTRY_FLOAT *)mat->NZ[i])->val;
        }
    }

    *ans = trace_sum;
    return 0;
}

/**
 * Performs matrix addition on mat1 and mat2 storing the answer in the ans
 * matrix. This is only well defined for matrices with the same dimensions.
 *
 * Returns non-zero on failure.
 */
int matrixAddition(COO *mat1, COO *mat2, COO *ans) {
    // quick dimension check
    if (mat1->cols != mat2->cols || mat1->rows != mat2->rows) return -1;

    // set up the answer matrix
    *ans = *mat1;
    ans->NZ = NULL;
    ans->nzsize = 0;

    int m1i = 0;
    int m2i = 0;
    if (mat1->type == MAT_FLOAT) {
        /* FLOAT ADDITION */
        while (1) {
            // create new entry
            COO_ENTRY_FLOAT *fl =
                (COO_ENTRY_FLOAT *)malloc(sizeof(COO_ENTRY_FLOAT));

            // row and column match -> add values
            if (mat1->NZ[m1i]->row == mat2->NZ[m2i]->row &&
                mat1->NZ[m1i]->col == mat2->NZ[m2i]->col) {
                // Perform addition
                *fl = *((COO_ENTRY_FLOAT *)mat1->NZ[m1i]);
                fl->val += ((COO_ENTRY_FLOAT *)mat2->NZ[m2i])->val;
                m1i++;
                m2i++;
            } else if (mat1->NZ[m1i]->row == mat2->NZ[m2i]->row) {
                // rows match
                if (mat1->NZ[m1i]->col < mat2->NZ[m2i]->col) {
                    // mat1 entry has a smaller position
                    *fl = *((COO_ENTRY_FLOAT *)mat1->NZ[m1i]);
                    m1i++;
                } else {
                    // mat2 entry has a smaller position
                    *fl = *((COO_ENTRY_FLOAT *)mat2->NZ[m2i]);
                    m2i++;
                }
            } else if (mat1->NZ[m1i]->row < mat2->NZ[m2i]->row) {
                // mat1 entry comes before mat2 regardless of col value
                *fl = *((COO_ENTRY_FLOAT *)mat1->NZ[m1i]);
                m1i++;
            } else {
                // mat2 entry comes before mat1 regardless of col value
                *fl = *((COO_ENTRY_FLOAT *)mat2->NZ[m2i]);
                m2i++;
            }

            // add new float entry to answer matrix
            ans->NZ =
                realloc(ans->NZ, (1 + ans->nzsize) * sizeof(COO_ENTRY_FLOAT));
            ans->NZ[ans->nzsize] = fl;
            ans->nzsize++;

            // break out of loop when we've finished comparing non-zero elements
            if (m1i == mat1->nzsize || m2i == mat2->nzsize) break;
        }

        // Fill rest of elements into answer matrix
        if (m1i == mat1->nzsize && m2i < mat2->nzsize) {
            // allocate enough memory for the rest of the elements
            ans->NZ = realloc(ans->NZ, (mat2->nzsize - ans->nzsize + 1) *
                                           sizeof(COO_ENTRY_FLOAT));

            int i;
#pragma omp parallel for
            for (i = m2i; i < mat2->nzsize; i++) {
                // create new entry
                COO_ENTRY_FLOAT *fl =
                    (COO_ENTRY_FLOAT *)malloc(sizeof(COO_ENTRY_FLOAT));
                *fl = *((COO_ENTRY_FLOAT *)mat2->NZ[i]);

                // add new float entry to answer matrix
                ans->NZ[ans->nzsize] = fl;
                ans->nzsize++;
            }
        } else if (m2i == mat2->nzsize && m1i < mat1->nzsize) {
            // allocate enough memory for the rest of the elements
            ans->NZ = realloc(ans->NZ, (mat1->nzsize - ans->nzsize + 1) *
                                           sizeof(COO_ENTRY_FLOAT));

            int i;
#pragma omp parallel for
            for (i = m1i; i < mat1->nzsize; i++) {
                // create new entry
                COO_ENTRY_FLOAT *fl =
                    (COO_ENTRY_FLOAT *)malloc(sizeof(COO_ENTRY_FLOAT));
                *fl = *((COO_ENTRY_FLOAT *)mat1->NZ[i]);

                // add new float entry to answer matrix
                ans->NZ[ans->nzsize] = fl;
                ans->nzsize++;
            }
        }
    } else {
        /* INTEGER ADDITION */
        while (1) {
            // create new entry
            COO_ENTRY_INT *fl = (COO_ENTRY_INT *)malloc(sizeof(COO_ENTRY_INT));

            // row and column match -> add values
            if (mat1->NZ[m1i]->row == mat2->NZ[m2i]->row &&
                mat1->NZ[m1i]->col == mat2->NZ[m2i]->col) {
                // Perform addition
                *fl = *((COO_ENTRY_INT *)mat1->NZ[m1i]);
                fl->val += ((COO_ENTRY_INT *)mat2->NZ[m2i])->val;
                m1i++;
                m2i++;
            } else if (mat1->NZ[m1i]->row == mat2->NZ[m2i]->row) {
                // rows match
                if (mat1->NZ[m1i]->col < mat2->NZ[m2i]->col) {
                    // mat1 entry has a smaller position
                    *fl = *((COO_ENTRY_INT *)mat1->NZ[m1i]);
                    m1i++;
                } else {
                    // mat2 entry has a smaller position
                    *fl = *((COO_ENTRY_INT *)mat2->NZ[m2i]);
                    m2i++;
                }
            } else if (mat1->NZ[m1i]->row < mat2->NZ[m2i]->row) {
                // mat1 entry comes before mat2 regardless of col value
                *fl = *((COO_ENTRY_INT *)mat1->NZ[m1i]);
                m1i++;
            } else {
                // mat2 entry comes before mat1 regardless of col value
                *fl = *((COO_ENTRY_INT *)mat2->NZ[m2i]);
                m2i++;
            }

            // add new float entry to answer matrix
            ans->NZ =
                realloc(ans->NZ, (1 + ans->nzsize) * sizeof(COO_ENTRY_INT));
            ans->NZ[ans->nzsize] = fl;
            ans->nzsize++;

            // break out of loop when we've finished comparing non-zero elements
            if (m1i == mat1->nzsize || m2i == mat2->nzsize) break;
        }

        // Fill rest of elements into answer matrix
        if (m1i == mat1->nzsize && m2i < mat2->nzsize) {
            // allocate enough memory for the rest of the elements
            ans->NZ = realloc(ans->NZ,
                              (mat2->nzsize - m2i + 1) * sizeof(COO_ENTRY_INT));
            int i;
#pragma omp parallel for
            for (i = m2i; i < mat2->nzsize; i++) {
                // create new entry
                COO_ENTRY_INT *fl =
                    (COO_ENTRY_INT *)malloc(sizeof(COO_ENTRY_INT));

                *fl = *((COO_ENTRY_INT *)mat2->NZ[i]);

                // add new float entry to answer matrix
                ans->NZ[ans->nzsize] = fl;
                ans->nzsize++;
            }
        } else if (m2i == mat2->nzsize && m1i < mat1->nzsize) {
            // allocate enough memory for the rest of the elements
            ans->NZ = realloc(ans->NZ,
                              (mat1->nzsize - m1i + 1) * sizeof(COO_ENTRY_INT));

            int i;
#pragma omp parallel for
            for (i = m1i; i < mat1->nzsize; i++) {
                // create new entry
                COO_ENTRY_INT *fl =
                    (COO_ENTRY_INT *)malloc(sizeof(COO_ENTRY_INT));

                *fl = *((COO_ENTRY_INT *)mat1->NZ[i]);

                // add new float entry to answer matrix
                ans->NZ[ans->nzsize] = fl;
                ans->nzsize++;
            }
        }
    }
    return 0;
}

/**
 * Transposes matrix mat and stores it in ans. Returns non-zero on failure.
 */
int matrixTranspose(COO *mat, COO *ans) {
    *ans = *mat;
    int i;

#pragma omp parallel for
    for (i = 0; i < ans->nzsize; i++) {
        if (ans->NZ[i]->row == ans->NZ[i]->col) continue;

        // swap row index for column index
        int temp = ans->NZ[i]->row;
        ans->NZ[i]->row = ans->NZ[i]->col;
        ans->NZ[i]->col = temp;
    }

    return 0;
}

/**
 * Performs matrix multiplication between mat1 and mat2 and stores the answer in
 * matrix ans. Returns non-zero on failure.
 */
int matrixMultiplication(CS *mat1, CS *mat2, CS *ans) {
    // check dimensions
    if (mat1->cols != mat2->rows) {
        return -1;
    }

    // set up the answer matrix
    ans->rows = mat1->rows;
    ans->cols = mat2->cols;
    ans->type = mat1->type;
    ans->nnzsize = 0;
    ans->NNZ = NULL;
    ans->JA = NULL;
    ans->IA = malloc((ans->rows + 1) * sizeof(int));

    if (ans->type == MAT_FLOAT) {
        for (int row = 0; row < ans->rows; row++) {
            int nnzCount = 0;

            // check if there are even any non-zeroes on this row
            if (mat1->IA[row + 1] - mat1->IA[row] > 0) {
                for (int col = 0; col < ans->cols; col++) {
                    float val = 0.0;
                    int r1 = mat1->IA[row];  // index of first non-zero on row
                    int c2 = mat2->IA[col];  // index of first non-zero on col

                    while (r1 < mat1->IA[row + 1] && c2 < mat2->IA[col + 1]) {
                        if (mat1->JA[r1] == mat2->JA[c2]) {
                            val += ((CS_ENTRY_FLOAT *)mat1->NNZ[r1])->val *
                                   ((CS_ENTRY_FLOAT *)mat2->NNZ[c2])->val;

                            r1++;
                            c2++;
                        } else if (mat1->JA[r1] < mat2->JA[c2]) {
                            // multiplying r1 with 0
                            r1++;
                        } else {
                            // multiplying c2 with 0
                            c2++;
                        }
                    }

                    if (val != 0) {
                        // this entry has a non-zero entry! create new CSR entry
                        CS_ENTRY_FLOAT *fl = malloc(sizeof(CS_ENTRY_FLOAT));
                        fl->val = val;

                        ans->NNZ =
                            realloc(ans->NNZ, (ans->nnzsize + 1) *
                                                  sizeof(CS_ENTRY_FLOAT));
                        ans->JA =
                            realloc(ans->JA, (ans->nnzsize + 1) * sizeof(int));

                        ans->NNZ[ans->nnzsize] = (CS_ENTRY_FLOAT *)fl;
                        ans->JA[ans->nnzsize] = col;

                        ans->nnzsize++;
                        nnzCount++;
                    }
                }
            }
            // update IA array
            ans->IA[row + 1] = ans->IA[row] + nnzCount;
        }
    } else {
        for (int row = 0; row < ans->rows; row++) {
            int nnzCount = 0;

            // check if there are even any non-zeroes on this row
            if (mat1->IA[row + 1] - mat1->IA[row] > 0) {
                for (int col = 0; col < ans->cols; col++) {
                    int val = 0;
                    int r1 = mat1->IA[row];  // index of first non-zero on row
                    int c2 = mat2->IA[col];  // index of first non-zero on col

                    while (r1 < mat1->IA[row + 1] && c2 < mat2->IA[col + 1]) {
                        if (mat1->JA[r1] == mat2->JA[c2]) {
                            val += ((CS_ENTRY_INT *)mat1->NNZ[r1])->val *
                                   ((CS_ENTRY_INT *)mat2->NNZ[c2])->val;

                            r1++;
                            c2++;
                        } else if (mat1->JA[r1] < mat2->JA[c2]) {
                            // multiplying r1 with 0
                            r1++;
                        } else {
                            // multiplying c2 with 0
                            c2++;
                        }
                    }

                    if (val != 0) {
                        // this entry has a non-zero entry! create new CSR entry
                        CS_ENTRY_INT *fl = malloc(sizeof(CS_ENTRY_INT));
                        fl->val = val;

                        ans->NNZ = realloc(ans->NNZ, (ans->nnzsize + 1) *
                                                         sizeof(CS_ENTRY_INT));
                        ans->JA =
                            realloc(ans->JA, (ans->nnzsize + 1) * sizeof(int));

                        ans->NNZ[ans->nnzsize] = (CS_ENTRY_BASE *)fl;
                        ans->JA[ans->nnzsize] = col;

                        ans->nnzsize++;
                        nnzCount++;
                    }
                }
            }
            // update IA array
            ans->IA[row + 1] = ans->IA[row] + nnzCount;
        }
    }

    return 0;
}
