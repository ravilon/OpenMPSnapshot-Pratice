#include <iostream>
#include <omp.h>
#include <chrono>
#include <cstdlib>
#include <vector>

using namespace std;

// Function to partition the array
int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return (i + 1);
}

// Serial Quick Sort
void serialQuickSort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        serialQuickSort(arr, low, pi - 1);
        serialQuickSort(arr, pi + 1, high);
    }
}

// Parallel Quick Sort
void parallelQuickSort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        #pragma omp task
        {
            parallelQuickSort(arr, low, pi - 1);
        }

        #pragma omp task
        {
            parallelQuickSort(arr, pi + 1, high);
        }
    }
}

int main() {
    vector<int> sizes = {1000, 2500, 8000, 20000, 100000};
    vector<float> durations_serial, durations_parallel;
	printf("Done by Dhivyesh R K 2021BCS0084\n");
    for (int N : sizes) {
        int arr[N], arr_copy[N];

        // Initialize the array with random values
        for (int i = 0; i < N; i++) {
            arr[i] = rand() % 100000;
            arr_copy[i] = arr[i]; // Make a copy for parallel sorting
        }

        // Serial Quick Sort
        auto start_time_serial = chrono::high_resolution_clock::now();
        serialQuickSort(arr, 0, N - 1);
        auto end_time_serial = chrono::high_resolution_clock::now();
        auto duration_serial = chrono::duration_cast<chrono::milliseconds>(end_time_serial - start_time_serial).count();
        durations_serial.push_back(static_cast<float>(duration_serial));

        // Parallel Quick Sort
        auto start_time_parallel = chrono::high_resolution_clock::now();
        #pragma omp parallel
        {
            #pragma omp single
            {
                parallelQuickSort(arr_copy, 0, N - 1);
            }
        }
        auto end_time_parallel = chrono::high_resolution_clock::now();
        auto duration_parallel = chrono::duration_cast<chrono::milliseconds>(end_time_parallel - start_time_parallel).count();
        durations_parallel.push_back(static_cast<float>(duration_parallel));
    }

    // Print durations
    for (size_t i = 0; i < sizes.size(); ++i) {
        printf("Size %d: Serial: %.9f ms, Parallel: %.9f ms\n", sizes[i], durations_serial[i], durations_parallel[i]);
    }

    return 0;
}

