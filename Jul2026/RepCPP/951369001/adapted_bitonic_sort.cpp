#include "adapted_bitonic_sort.h"


/**
 * Adapted bitonic sort.
 *
 * @details work even if the array.size() is not divisible by 2^k.
 *
 * @param array
 */
void Adapted_Bitonic_Sort::adapted_bitonic_sort(std::vector<int> &array) {
    sort_adapted_bitonic(array, 0, static_cast<int>(array.size()), ASCENDING);
}


/**
 * The sort algorithm of adapted bitonic sort.
 *
 * @param array the unsorted array.
 * @param start_position the starting position.
 * @param array_size the array size.
 * @param direction the sorting direction.
 */
void Adapted_Bitonic_Sort::sort_adapted_bitonic(std::vector<int> &array, const int start_position, const int array_size, const bool direction) {
    if(array_size > 1) {
        ///The subarray size.
        const int subarray_size = array_size / 2;


        #pragma omp parallel sections
        {
            #pragma omp section
            {
                sort_adapted_bitonic(array, start_position, subarray_size, !direction);
            }

            #pragma omp section
            {
                sort_adapted_bitonic(array, start_position + subarray_size, array_size - subarray_size, direction);
            }
        }

        merge_adapted_bitonic(array, start_position, array_size, direction);
    }
}

/**
 * The merge algorithm of adapted bitonic sort.
 *
 * @param array the unsorted array.
 * @param start_position the starting position.
 * @param array_size the array size.
 * @param direction the sorting direction.
 */
void Adapted_Bitonic_Sort::merge_adapted_bitonic(std::vector<int> &array, const int start_position, const int array_size, const bool direction) {
    if(array_size > 1) {
        ///The subarraysize.
        const int subarray_size = greatest_power_of_2_less_than(array_size);


        #pragma omp parallel for
        for (int i = start_position; i < start_position + array_size - subarray_size; ++i) {
            compare_and_swap(array, i, i + subarray_size, direction);
        }

        merge_adapted_bitonic(array, start_position, subarray_size, direction);
        merge_adapted_bitonic(array, start_position + subarray_size, array_size - subarray_size, direction);
    }
}

/**
 * Function that compares and swaps two elements based on the sorting direction.
 *
 * @param array the unsorted array.
 * @param first_index the starting position.
 * @param second_index the finishing position.
 * @param direction the sorting direction.
 */
void Adapted_Bitonic_Sort::compare_and_swap(std::vector<int> &array, const int first_index, const int second_index, const bool direction) {
    if(direction == array[first_index] > array[second_index]) {
        std::swap(array[first_index], array[second_index]);
    }
}

/**
 * Function that finds the greatest power of two that is less than or equal to a given number.
 *
 * @param n the input number.
 * @return the greatest power of two less than or equal to the input number.
 */
int Adapted_Bitonic_Sort::greatest_power_of_2_less_than(const int n) {
    ///The greatest power of two less or equal than the input number.
    int k = 1;


    while (k > 0 && k < n) {
        k = k << 1;
    }

    return k >> 1;
}
