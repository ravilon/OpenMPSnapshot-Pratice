// Course:           High Performance Computing
// A.Y:              2021/22
// Lecturer:         Francesco Moscato           fmoscato@unisa.it

// Team:
// Alessio Pepe          0622701463      a.pepe108@studenti.unisa.it
// Teresa Tortorella     0622701507      t.tortorella3@studenti.unisa.it
// Paolo Mansi           0622701542      p.mansi5@studenti.unisa.it

// Copyright (C) 2021 - All Rights Reserved

// This file is part of Counting_Sort.

// Counting_Sort is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// Counting_Sort is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with Counting_Sort.  If not, see <http://www.gnu.org/licenses/>.

/**
 * @file    counting_sort.c
 * @author  Alessio Pepe         (a.pepe108@studenti.unisa.it)
 * @author  Paolo Mansi          (p.mansi5@studenti.unisa.it)
 * @author  Teresa Tortorella    (t.tortorella3@studenti.unisa.it)
 * @brief   Counting sort alghorithm in serial and parallelized (OpenMP) version. Serial
 *          version are inspired at https://it.wikipedia.org/wiki/Counting_sort.   
 * @version 1.0.0
 * @date 2021-11-01
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include <stdlib.h>
#include "counting_sort.h"
#include "util.h"

/**
 * @brief If defined, use the first minmax alghorithm described in the report.
 * 
 */
#define MINMAX_1

/**
 * @brief If defined, use the first count_occurrance alghorithm described in the report.
 * 
 */
#define COUNT_OCCURRANCE_1

/**
 * @brief If defined, use the second count_occurrance alghorithm described in the report.
 * 
 */
//#define COUNT_OCCURRANCE_2

/**
 * @brief If defined, use the first populate alghorithm described in the report.
 * 
 */
#define POPULATE_1

/**
 * @brief If defined, use the second populate alghorithm described in the report.
 * 
 */
//#define POPULATE_2


/**
 * @brief   Reorder the array using a counter-sort alghorithm.
 * 
 * @param A     The pointer to the array to reorder.
 * @param A_len   The lenght to the array to reorder.
 */
void counting_sort(ELEMENT_TYPE *A, size_t A_len) 
{
    // Empty or lenght=1 array
    if (A_len < 2)
    {
        return;
    }

    ELEMENT_TYPE min, max;
    size_t *C;
    size_t C_len, k;

    //Compute max and min of v.
    min = A[0];
    max = A[0];

    for (size_t i = 1; i < A_len; i++)
    {
        if (A[i] < min)
        {
            min = A[i];
        }
        else if (A[i] > max)
        {
            max = A[i];
        }
    }

    // Construct a zeros array of lenght max-min+1.
    C_len = max - min + 1;
    C = (size_t *) calloc(C_len, sizeof(size_t));

    // Count the element frequency.
    for (size_t i = 0; i < A_len; i++)
    {
        C[A[i] - min]++;
    }
    
    // Ordering based on the frequency array.
    k = 0;

    for (size_t i = 0; i < C_len; i++)
    {
        for (size_t j = 0; j < C[i]; j++)
        {
            A[k++] = i + min;
        }
    }
    
    free(C);    // Dealloc the frequency vector.
}

/**
 * @brief   Reorder the array using a counter-sort alghorithm parallelized.
 * 
 * @param v             The pointer to the array to reorder.
 * @param A_len           The lenght to the array to reorder.
 * @param threads       The thread to use to run this function. If you want use 
 *                      default number use 0.
 */
void counting_sort_parall1(ELEMENT_TYPE *A, size_t A_len, int threads) 
{
    // Empty or lenght=1 array
    if (A_len < 2)
    {
        return;
    }

    ELEMENT_TYPE min, max;
    size_t *C;
    size_t C_len;

#ifdef MINMAX_1
    // Compute min and max
    min = A[0];
    max = A[0];

    #pragma omp parallel default(none) firstprivate(A_len, A) shared(max, min) num_threads(threads) 
    {
        // We made a local variable for max and min for each thread
        ELEMENT_TYPE l_min = A[0];
        ELEMENT_TYPE l_max = A[0];

        // Each thread compute max and min on his part of the array. That don't need
        // to wait because the next part depends just on the local min and max of the thread.
        #pragma omp for nowait
        for (size_t i = 1; i < A_len; i++)
        {
            if (A[i] < l_min)
            {
                l_min = A[i];
            }
            else if (A[i] > l_max)
            {
                l_max = A[i];
            }
        }

        // Each thread update the global min and max.
        #pragma omp critical
        {
            if (l_min < min)
            {
                min = l_min;
            }
            if (l_max > max)
            {
                max = l_max;
            }
        }
    }
#endif

    // Construct a zeros array of lenght max-min+1.
    C_len = max - min + 1;
    C = (size_t *) calloc(C_len, sizeof(size_t));

#if defined(COUNT_OCCURRANCE_1)
    // Count the element frequency.
    #pragma omp parallel default(none) firstprivate(A, A_len, C_len, min) shared(C) num_threads(threads)
    {
        // Each thread have a local C array inizialized to 0.
        size_t *C_loc = (size_t *) calloc(C_len, sizeof(size_t));

        // Each thread count frequency on a part of the array. They do not need to wait
        // the barrier. Otherwise thay can continue because they work only on their local
        // variable scope.
        #pragma omp for nowait
        for (size_t j = 0; j < A_len; j++)
        {
            C_loc[A[j] - min]++;
        }

        // Just one thread for time store his results in the global C array.
        #pragma omp critical
        {
            for (size_t k = 0; k < C_len; k++)
            {
                C[k] += C_loc[k];
            }
        }
        
        // Each thread dealloc him local temporary vector
        free(C_loc);
    }
#elif defined(COUNT_OCCURRANCE_2)
    #pragma omp parallel for default(none) shared(C) firstprivate(A, A_len, C_len, min) num_threads(threads)
    for (size_t i = 0; i < A_len; i++)
    {   
        #pragma omp atomic
        C[A[i] - min]++;
    }
#endif

#if defined(POPULATE_1)  
    // Compose an index based locator to populate the vector v.
    for (size_t i = 1; i < C_len; i++)
    {
        C[i] += C[i-1];
    }
    
    // Ordering based on the frequency array.
    #pragma omp parallel for default(none) firstprivate(A, C, C_len, min) num_threads(threads)
    for (size_t i = 0; i < C_len; i++)
    {
        size_t start = (i != 0 ? C[i-1] : 0);
        for (size_t j = start; j < C[i]; j++)
        {
            A[j] = i + min;
        }
    }
#elif defined(POPULATE_2)  
    // Compose an index based locator to populate the vector v.
    #pragma omp parallel for ordered default(none) firstprivate(C_len) shared(C) num_threads(threads)
    for (size_t i = 1; i < C_len; i++)
    {
        #pragma omp ordered
        {
            C[i] += C[i-1];
        }
    }
    
    // Ordering based on the frequency array.
    #pragma omp parallel for default(none) firstprivate(A, C, C_len, min) num_threads(threads)
    for (size_t i = 0; i < C_len; i++)
    {
        size_t start = (i != 0 ? C[i-1] : 0);
        for (size_t j = start; j < C[i]; j++)
        {
            A[j] = i + min;
        }
    }
#endif
    
    // Dealloc the frequency vector.
    free(C); 
}

