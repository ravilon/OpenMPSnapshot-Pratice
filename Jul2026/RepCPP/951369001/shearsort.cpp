#include "shearsort.h"


/**
 * Shearsort.
 *
 * @warning work only with a squared matrix.
 *
 * @details the sorted matrix is sorted in snake direction.
 *
 * @param matrix the unsorted matrix.
 */
void Shearsort::shearSort(vector<vector<int>>& matrix) {
    ///The matrix size.
    const int matrix_size = static_cast<int>(matrix.size());
    ///The logarithm of the matrix size.
    const int logn = static_cast<int>(log2(matrix_size));


    for (int i = 0; i < logn; i++) {
        //sort rows
        sort_rows(matrix);
        //sort columns
        sort_columns(matrix);
    }

    //last sort rows
    sort_rows(matrix);
}


/**
 * Function that sorts the rows.
 *
 * @details sorting direction: even rows from left to right and odd rows from right to left.
 *
 * @param matrix the unsorted matrix.
 */
void Shearsort::sort_rows(vector<vector<int>>& matrix) {
    ///The matrix size.
    const int matrix_size = static_cast<int>(matrix.size());

    #pragma omp parallel for
    for (int i = 0; i < matrix_size; i++) {
        if (i % 2 == 0) { //even rows in ascending
            sort(matrix[i].begin(), matrix[i].end());
        } else { //odd rows in descending
            sort(matrix[i].rbegin(), matrix[i].rend());
        }
    }
}

/**
 * Function that sorts the columns.
 *
 * @param matrix the unsorted matrix.
 */
void Shearsort::sort_columns(vector<vector<int>>& matrix) {
    ///The matrix size.
    const int matrix_size = static_cast<int>(matrix.size());


    #pragma omp parallel for
    for (int j = 0; j < matrix_size; j++) {
        ///The column.
        vector<int> column(matrix_size);

        for (int i = 0; i < matrix_size; i++) {
            column[i] = matrix[i][j];
        }

        sort(column.begin(), column.end());

        for (int i = 0; i < matrix_size; i++) {
            matrix[i][j] = column[i];
        }
    }
}
