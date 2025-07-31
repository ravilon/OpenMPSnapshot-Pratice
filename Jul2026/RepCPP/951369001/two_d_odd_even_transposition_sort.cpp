#include "two_d_odd_even_transposition_sort.h"


/**
 * 2D odd-even transposition sort.
 *
 * @warning work only with a squared matrix.
 *
 * @details the sorted matrix is sorted in snake direction.
 *
 * @param matrix the unsorted matrix.
 */
void Two_D_Odd_Even_Transposition_Sort::two_d_odd_even_sort(vector<vector<int>>& matrix) {
    ///The matrix is sorted or not.
    bool is_sorted = true;


    while (is_sorted) {
        is_sorted = false;

        //odd oets step to the rows
        is_sorted |= sort_row_oets_step(matrix, true);
        //even oets step to the rows
        is_sorted |= sort_row_oets_step(matrix, false);
        //odd oets step to the columns
        is_sorted |= sort_column_oets_step(matrix, true);
        //even oets step to the columns
        is_sorted |= sort_column_oets_step(matrix, false);
    }
}


/**
 * Function that sorts a row following the specified direction.
 *
 * @details sorting direction: even rows from left to right and odd rows from right to left.
 *
 * @param matrix the unsorted matrix.
 * @param is_odd the oets step on the rows is either odd or even, true if is odd.
 * @return true if the column is sorted, false otherwise.
 */
bool Two_D_Odd_Even_Transposition_Sort::sort_row_oets_step(vector<vector<int>> &matrix, bool is_odd) {
    ///The matrix size.
    const int n = static_cast<int>(matrix.size());
    ///The array is sorted or not.
    bool is_sorted_rows = false;
    ///The sorting direction of the rows.
    bool direction = false;


    #pragma omp parallel for reduction(|:is_sorted_rows) default(none) shared(matrix, n, is_odd) private(direction)
    for (int i = 0; i < n; ++i) {
        direction = i % 2 == 0;

        for (int j = is_odd ? 1 : 0; j < n - 1; j += 2) {
            ///The index of the first value to be swapped.
            int a = j;
            ///The index of the second value to be swapped.
            int b = j + 1;


            if (!direction) {
                swap(a, b);
            }

            {
                if (matrix[i][a] > matrix[i][b]) {
                    swap(matrix[i][a], matrix[i][b]);
                    is_sorted_rows = true;
                }
            }
        }
    }

    return is_sorted_rows;
}


/**
 * Function that sorts a column.
 *
 * @param matrix the unsorted matrix.
 * @param is_odd the oets step on the rows is either odd or even, true if is odd.
 * @return true if the column is sorted, false otherwise.
 */
bool Two_D_Odd_Even_Transposition_Sort::sort_column_oets_step(vector<vector<int>> &matrix, bool is_odd) {
    ///The matrix size.
    const int n = static_cast<int>(matrix.size());
    ///The columns is sorted or not.
    bool is_sorted_columns = false;


    #pragma omp parallel for reduction(|:is_sorted_columns) default(none) shared(matrix, n, is_odd)
    for (int i = 0; i < n; ++i) {
        for (int j = (is_odd ? 1 : 0); j < n - 1; j += 2) {

            #pragma omp critical
            {
                if (matrix[j][i] > matrix[j + 1][i]) {
                    swap(matrix[j][i], matrix[j + 1][i]);
                    is_sorted_columns = true;
                }
            }
        }
    }

    return is_sorted_columns;
}

