#include "bitonic_sort.h"


/**
 * Bitonic Sort
 *
 * @warning work only with array.size() divisible by 2^k.
 *
 * @param array the unsorted array.
 */
void Bitonic_Sort::bitonic_sort(std::vector<int>& array) {
    sort_bitonic(array, 0, static_cast<int>(array.size()), ASCENDING);
}


/**
 * The sort algorithm of bitonic sort.
 *
 * @param array the unsorted array.
 * @param start_position the starting position.
 * @param array_size the array size.
 * @param direction the sorting direction.
 */
void Bitonic_Sort::sort_bitonic(std::vector<int> &array, const int start_position, const int array_size, const bool direction) {
    if(array_size > 1) {
        ///The subarray size.
        const int subarray_size = array_size / 2;


        #pragma omp parallel sections
        {
            #pragma omp section
            {
                sort_bitonic(array, start_position, subarray_size, ASCENDING);
            }

            #pragma omp section
            {
                sort_bitonic(array, start_position + subarray_size, subarray_size, DESCENDING);
            }
        }

        merge_bitonic(array, start_position, array_size, direction);
    }
}

/**
 * The merge algorithm of bitonic sort.
 *
 * @param array the unsorted array.
 * @param start_position the starting position.
 * @param array_size the array size.
 * @param direction the sorting direction.
 */
void Bitonic_Sort::merge_bitonic(std::vector<int> &array, const int start_position, const int array_size, const bool direction) {
    if(array_size > 1) {
        ///The subarray size.
        const int subarray_size = array_size / 2;


        #pragma omp parallel for
        for (int i = start_position; i < start_position + subarray_size; ++i) {
            compare_and_swap(array, i, i + subarray_size, direction);
        }

        merge_bitonic(array, start_position, subarray_size, direction);
        merge_bitonic(array, start_position + subarray_size, subarray_size, direction);
    }
}

/**
 * Function that compares and swaps two elements based on the sorting direction.
 *
 * @param array the unsorted array.
 * @param first_index the index of the first element to be swapped.
 * @param second_index the index of the second element to be swapped.
 * @param direction the sorting direction.
 */
void Bitonic_Sort::compare_and_swap(vector<int> &array, const int first_index, const int second_index, const bool direction) {
    if(direction == array[first_index] > array[second_index]) {
        swap(array[first_index], array[second_index]);
    }
}