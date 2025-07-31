#include "odd_even_mergesort.h"


/**
 * Odd-even mergesort algorithm.
 *
 * @warning work only with array.size() divisible by 2^k.
 *
 * @param array the unsorted array.
 */
void Odd_Even_Mergesort::odd_even_mergesort(vector<int>& array) {
    mergesort_odd_even(array, 0, static_cast<int>(array.size()));
}


/**
 * The mergesort algorithm of odd-even mergesort.
 *
 * @param array the unsorted array.
 * @param start_position the starting position.
 * @param finish_position the finishing position.
 */
void Odd_Even_Mergesort::mergesort_odd_even(vector<int>& array, const int start_position, const int finish_position) {
    if (finish_position > 1) {
        ///The midpoint.
        const int midpoint = finish_position / 2;


        #pragma omp parallel sections
        {
            #pragma omp section
            {
                mergesort_odd_even(array, start_position, midpoint);
            }

            #pragma omp section
            {
                mergesort_odd_even(array, start_position + midpoint, midpoint);
            }
        }

        merge_odd_even(array, start_position, finish_position, 1);
    }
}

/**
 * The merge algorithm of odd-even mergesort.
 *
 * @param array the unsorted array.
 * @param start_position the starting position.
 * @param finish_position the finishing position.
 * @param distance_to_compare the distance of the elements to be compared
 */
void Odd_Even_Mergesort::merge_odd_even(vector<int>& array, const int start_position, const int finish_position, const int distance_to_compare) {
    if (const int m = distance_to_compare * 2; m < finish_position) {
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                merge_odd_even(array, start_position, finish_position, m);
            }

            #pragma omp section
            {
                merge_odd_even(array, start_position + distance_to_compare, finish_position, m);
            }
        }

        #pragma omp parallel for
        for (int i = start_position + distance_to_compare; i < start_position + finish_position - distance_to_compare; i += m) {
            compare_and_swap(array, i, i + distance_to_compare);
        }
    }
    else {
        compare_and_swap(array, start_position, start_position + distance_to_compare);
    }
}

/**
 * The function compares and swaps two elements when the first is greater than the second.
 *
 * @param array the unsorted array.
 * @param first_index the index of the first element to be swapped.
 * @param second_index the index of the second element to be swapped.
 */
void Odd_Even_Mergesort::compare_and_swap(vector<int>& array, const int first_index, const int second_index) {
    if (array[first_index] > array[second_index]) {
        swap(array[first_index], array[second_index]);
    }
}
