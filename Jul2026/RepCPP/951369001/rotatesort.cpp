#include "rotatesort.h"


/**
 * Rotatesort.
 *
 * @warning work only with a squared matrix.
 *
 * @details the sorted matrix is sorted in snake direction.
 *
 * @param matrix the unsorted matrix.
 */
void Rotatesort::rotatesort(vector<vector<int>> &matrix) {
    ///The matrix size.
    const int n = static_cast<int>(matrix.size());


    //balance each vertical slice
    balance(matrix);;

    //unblock
    unblock(matrix);

    //balance each horizontal slice - transpose, balance, transpose back
    ///The transposed matrix.
    vector transposed(n, vector<int>(n));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }
    balance(transposed);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = transposed[j][i];
        }
    }

    //unblock
    unblock(matrix);

    //shear 3 times
    shear(matrix);
    shear(matrix);
    shear(matrix);

    //sort the last unsorted row
    sort(matrix[0].begin(), matrix[0].end());
}


/**
 * The balance operation of rotatesort.
 *
 * @param matrix the matrix.
 */
void Rotatesort::balance(vector<vector<int>> &matrix) {
    ///The matrix size.
    const int n = static_cast<int>(matrix.size());
    ///The square root of the matrix size.
    const int sqrt_n = static_cast<int>(sqrt(n));


    #pragma omp parallel for
    for (int slice_i = 0; slice_i < sqrt_n; slice_i++) {
        ///The slice.
        vector slice(n, vector<int>(sqrt_n));

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < sqrt_n; ++j) {
                slice[i][j] = matrix[i][slice_i * sqrt_n + j]; //{n x sqrt(n)}
            }
        }

        //sort columns
        sort_columns(slice);
        //rotate rows
        rotate_rows_balance(slice);
        //sort columns
        sort_columns(slice);

        //insert slice back
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < sqrt_n; ++j) {
                matrix[i][slice_i * sqrt_n + j] = slice[i][j];
            }
        }
    }
}

/**
 * The unblock operation of rotatesort.
 * 
 * @param matrix the unsorted matrix.
 */
void Rotatesort::unblock(vector<vector<int>> &matrix) {
    //rotate rows
    rotate_rows_unblock(matrix);
    //sort columns
    sort_columns(matrix);
}

/**
 * The shear operation of rotatesort.
 *
 * @param matrix the unsorted array.
 */
void Rotatesort::shear(vector<vector<int>> &matrix) {
    ///The matrix size.
    const int n = static_cast<int>(matrix.size());


    //sort the rows in alternating direction
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        if (i % 2 == 0) { //even rows in ascending
            sort(matrix[i].begin(), matrix[i].end());
        } else { //odd rows in descending
            sort(matrix[i].rbegin(), matrix[i].rend());
        }
    }

    //sort columns
    sort_columns(matrix);
}


/**
 * Function that sorts the columns.
 *
 * @param matrix the unsorted matrix.
 */
void Rotatesort::sort_columns(vector<vector<int>> &matrix) {
    ///The matrix size.
    const int n = static_cast<int>(matrix.size());


    #pragma omp parallel for
    for (int col = 0; col < n; ++col) {
        ///The column.
        vector<int> temp(n);

        for (int row = 0; row < n; ++row) {
            temp[row] = matrix[row][col];
        }
        sort(temp.begin(), temp.end());
        for (int row = 0; row < n; ++row) {
            matrix[row][col] = temp[row];
        }
    }
}

/**
 * Function used in balance operation that rotates the rows.
 *
 * @details rotate row i by {i % sqrt(n)} positions.
 *
 * @param matrix the unsorted matrix.
 */
void Rotatesort::rotate_rows_balance(vector<vector<int>> &matrix) {
    ///The matrix size.
    const int n = static_cast<int>(matrix.size());
    ///The square root of the matrix size.
    const int sqrt_n = static_cast<int>(sqrt(n));


    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        //rotate row i by (i % sqrt(n)) positions
        rotate(matrix[i].begin(), matrix[i].begin() + (i % sqrt_n), matrix[i].end());
    }
}

/**
 * Function used in unblock operation that rotates the rows.
 *
 * @details rotate row i by {(i * sqrt(n)) % n} positions.
 *
 * @param matrix the unsorted matrix.
 */
void Rotatesort::rotate_rows_unblock(vector<vector<int>> &matrix) {
    ///The matrix size.
    const int n = static_cast<int>(matrix.size());
    ///The square root of the matrix size.
    const int sqrt_n = static_cast<int>(sqrt(n));


    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        //rotate row i by (i * sqrt(n)) % n positions (used in unblock)
        rotate(matrix[i].begin(), matrix[i].begin() + ((i * sqrt_n) % n), matrix[i].end());
    }
}
