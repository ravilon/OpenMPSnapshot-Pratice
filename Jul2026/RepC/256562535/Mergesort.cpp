#include <iostream>
#include <omp.h>

#include "Mergesort.h"

using namespace std;

void Mergesort::merge(int32_t *arr, int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    /* create temp arrays */
    int32_t *L = (int32_t *)malloc(sizeof(int32_t) * n1);
    int32_t *R = (int32_t *)malloc(sizeof(int32_t) * n2);

    /* Copy data to temp arrays L[] and R[] */
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    /* Merge the temp arrays back into arr[l..r]*/
    i = 0; // Initial index of first subarray
    j = 0; // Initial index of second subarray
    k = l; // Initial index of merged subarray
    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
        {
            arr[k] = L[i];
            i++;
        }
        else
        {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    /* Copy the remaining elements of L[], if there 
       are any */
    while (i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }

    /* Copy the remaining elements of R[], if there 
       are any */
    while (j < n2)
    {
        arr[k] = R[j];
        j++;
        k++;
    }

    free(L);
    free(R);
}

void Mergesort::mergeSort(int32_t *arr, int l, int r)
{
    if (l < r)
    {
        // Same as (l+r)/2, but avoids overflow for large l and h
        int m = l + (r - l) / 2;

        // No taskwait is need
        #pragma omp taskgroup
        {
            // Avoid small chunks with omp (https://stackoverflow.com/a/47495419)
            #pragma omp task shared(arr) if (r - l >= MIN_MERGESORT_SIZE)
            mergeSort(arr, l, m);

            #pragma omp task shared(arr) if (r - l >= MIN_MERGESORT_SIZE)
            mergeSort(arr, m + 1, r);
        }
        merge(arr, l, m, r);
    }
}

void Mergesort::sort(int32_t *arr, int size)
{
    mergeSort(arr, 0, size - 1);
}

void Mergesort::sortParallel(int32_t *arr, int size)
{
    #pragma omp parallel
    #pragma omp single
    {
        cout << "Sorting with " << omp_get_num_threads() << " Threads ..." << endl;
        mergeSort(arr, 0, size - 1);
    }
}

void Mergesort::fillWithRandomNumbers(int32_t *arr, int size)
{
    unsigned int seed = time(NULL);

    for (int i = 0; i < size; ++i)
    {
        arr[i] = rand_r(&seed) % size;
    }
}

void Mergesort::print(const int32_t *arr, int size)
{
    if (size <= 0)
        return;

    cout << "[" << arr[0];

    if (size <= MAX_PRINT_SIZE)
    {
        // Print array
        for (int i = 1; i < size; ++i)
        {
            cout << ", " << arr[i];
        }
    }
    else
    {
        // Print fist and last part of a array
        int i;
        for (i = 1; i < MAX_PRINT_SIZE / 2; ++i)
        {
            cout << ", " << arr[i];
        }
        cout << ", ...";
        for (i = size - i; i < size; ++i)
        {
            cout << ", " << arr[i];
        }
    }
    cout << "]" << endl;
}
