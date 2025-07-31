#include "three_n_sort_Schnorr_and_Shamir.h"


/**
 * 3n-sort of Schnorr and Shamir.
 *
 * @warning work only with a squared matrix.
 *
 * @details the sorted matrix is sorted in snake direction.
 *
 * @param matrix the unsorted matrix.
 */
void Three_N_Sort_Schnorr_and_Shamir::three_n_sort(vector<vector<int>> &matrix) {
    ///The matrix size.
    const int n = static_cast<int>(matrix.size());


    //sort blocks
    sort_blocks(matrix); //n^{3/4}
    //{n^{1/4}}-way unshuffle along the rows
    k_way_unshuffle(matrix, static_cast<int>(ceil(pow(n, 1.0 / 4.0)))); //n^{1/4} on rows
    //sort blocks
    sort_blocks(matrix); //n^{3/4}
    //sort columns
    sort_columns(matrix);
    //sort vertical slices
    sort_vertical_slices(matrix); //n^{3/4}
    //sort rows
    sort_rows_alternating_direction(matrix);
    //n^{3/4} steps of oets
    odd_even_transposition_sort_snake(matrix); //n^{3/4} steps
}


/**
 * k-way unshuffle operation of 3n-sort of Schnorr and Shamir.
 *
 * @param matrix the unsorted matrix.
 * @param k the number of unshuffle way.
 */
void Three_N_Sort_Schnorr_and_Shamir::k_way_unshuffle(vector<vector<int>> &matrix, const int k) {
    ///The matrix size.
    const int n = static_cast<int>(matrix.size());
    ///The block size.
    const int block_size = n / k;
    ///The permutated matrix.
    vector temp(n, vector<int>(n));


    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            temp[i][j % k * block_size + j / k] = matrix[i][j];
        }
    }

    matrix = move(temp);
}


/**
 * Function that sorts the blocks of a matrix.
 *
 * @param matrix the unsorted matrix.
 * @param block_size the block size.
 */
void Three_N_Sort_Schnorr_and_Shamir::sort_blocks(vector<vector<int>> &matrix) {
    ///The matrix size.
    const int n = static_cast<int>(matrix.size());
    ///The block size.
    const int block_size = static_cast<int>(pow(n, 3.0 / 4.0));


    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < n; i += block_size) {
        for (int j = 0; j < n; j += block_size) {
            ///The temporary block.
            vector<int> temp(block_size * block_size);
            ///The index in the block.
            int index = 0;

            for (int bi = 0; bi < block_size; ++bi) {
                for (int bj = 0; bj < block_size; ++bj) {
                    temp[index++] = matrix[i + bi][j + bj];
                }
            }

            sort(temp.begin(), temp.end());

            index = 0;
            for (int bi = 0; bi < block_size; ++bi) {
                for (int bj = 0; bj < block_size; ++bj) {
                    matrix[i + bi][j + bj] = temp[index++];
                }
            }
        }
    }
}

/**
 * Function that sorts the columns of a matrix.
 *
 * @param matrix the unsorted matrix.
 */
void Three_N_Sort_Schnorr_and_Shamir::sort_columns(vector<vector<int>> &matrix) {
    ////The matrix size.
    const int n = static_cast<int>(matrix.size());


    #pragma omp parallel for
    for (int j = 0; j < n; ++j) {
        ///The column.
        vector<int> column(n);

        for (int i = 0; i < n; ++i) {
            column[i] = matrix[i][j];
        }

        sort(column.begin(), column.end());

        for (int i = 0; i < n; ++i) {
            matrix[i][j] = column[i];
        }
    }
}

/**
 * Function that sorts the vertical slices.
 *
 * @param matrix the unsorted matrix.
 * @param slice_width the slice width.
 */
void Three_N_Sort_Schnorr_and_Shamir::sort_vertical_slices(vector<vector<int>> &matrix) {
    ///The matrix size.
    const int n = static_cast<int>(matrix.size());
    ///The slice width.
    const int slice_width = static_cast<int>(pow(n, 3.0 / 4.0));


    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < n; j += slice_width) {
        ///The slice.
        vector<int> temp;
        temp.reserve(n * slice_width);

        for (int i = 0; i < n; ++i) {
            for (int sj = 0; sj < slice_width && j + sj < n; ++sj) {
                temp.push_back(matrix[i][j + sj]);
            }
        }

        sort(temp.begin(), temp.end());

        ///The index in the slice.
        int index = 0;

        for (int i = 0; i < n; ++i) {
            for (int sj = 0; sj < slice_width && j + sj < n; ++sj) {
                matrix[i][j + sj] = temp[index++];
            }
        }
    }
}

/**
 * Function that sorts the rows in alternating direction.
 *
 * @param matrix the unsorted matrix.
 */
void Three_N_Sort_Schnorr_and_Shamir::sort_rows_alternating_direction(vector<vector<int>> &matrix) {
    ///The matrix size.
    const int n = static_cast<int>(matrix.size());


    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        if (i % 2 == 0) { //sorted in ascending order
            sort(matrix[i].begin(), matrix[i].end());
        }
        else { //sorted in descending order
            sort(matrix[i].rbegin(), matrix[i].rend());
        }
    }
}

/**
 * Function that executes the odd-even transposition sort to the snake.
 *
 * @param matrix the unsorted matrix.
 */
void Three_N_Sort_Schnorr_and_Shamir::odd_even_transposition_sort_snake(vector<vector<int>> &matrix) {
    ///The matrix size.
    const int n = static_cast<int>(matrix.size());
    ///The number of odd-even transposition steps.
    const int steps = static_cast<int>(round(pow(n, 3.0 / 4.0))); //only n^{3/4} steps
    ///The array.
    vector<int> snake;


    for (int i = 0; i < n; ++i) {
        if (i % 2 == 0) { //sorted in ascending order
            snake.insert(snake.end(), matrix[i].begin(), matrix[i].end());
        }
        else { //sorted in descending order
            snake.insert(snake.end(), matrix[i].rbegin(), matrix[i].rend());
        }
    }

    //partial odd-even transposition sort (n^{3/4} steps)
    for (int step = 0; step < steps; ++step) {
        #pragma omp parallel for
        for (int i = step % 2; i < n * n - 1; i += 2) {
            if (snake[i] > snake[i + 1]) {
                swap(snake[i], snake[i + 1]);
            }
        }
    }

    ///The index in the array.
    int index = 0;

    for (int i = 0; i < n; ++i) {
        if (i % 2 == 0) {
            copy_n(snake.begin() + index, n, matrix[i].begin());
        }
        else {
            copy_n(snake.begin() + index, n, matrix[i].rbegin());
        }
        index += n;
    }
}
