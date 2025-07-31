#include <iostream>
#include <vector>
#include <omp.h>
#include <random>
#include <chrono>
using namespace std;

// Normal merge sort
void merge(vector<int>& arr, int left, int middle, int right) {
    int n1 = middle - left + 1;
    int n2 = right - middle;

    vector<int> leftArr(n1);
    vector<int> rightArr(n2);

    for (int i = 0; i < n1; i++)
        leftArr[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        rightArr[j] = arr[middle + 1 + j];

    int i = 0, j = 0, k = left;

    while (i < n1 && j < n2) {
        if (leftArr[i] <= rightArr[j]) {
            arr[k] = leftArr[i];
            i++;
        } else {
            arr[k] = rightArr[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = leftArr[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = rightArr[j];
        j++;
        k++;
    }
}

void serialMergeSort(vector<int>& arr, int left, int right) {
    if (left < right) {
        int middle = left + (right - left) / 2;

        serialMergeSort(arr, left, middle);
        serialMergeSort(arr, middle + 1, right);

        merge(arr, left, middle, right);
    }
}

// Parallel merge sort
void parallelMerge(vector<int>& arr, int left, int middle, int right) {
    int n1 = middle - left + 1;
    int n2 = right - middle;

    vector<int> leftArr(n1);
    vector<int> rightArr(n2);

    #pragma omp parallel for
    for (int i = 0; i < n1; i++)
        leftArr[i] = arr[left + i];

    #pragma omp parallel for
    for (int j = 0; j < n2; j++)
        rightArr[j] = arr[middle + 1 + j];

    int i = 0, j = 0, k = left;

    while (i < n1 && j < n2) {
        if (leftArr[i] <= rightArr[j]) {
            arr[k] = leftArr[i];
            i++;
        } else {
            arr[k] = rightArr[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = leftArr[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = rightArr[j];
        j++;
        k++;
    }
}

void parallelMergeSort(vector<int>& arr, int left, int right, int depth) {
    if (left < right) {
        int middle = left + (right - left) / 2;

            #pragma omp parallel sections
            {
                #pragma omp section
                {
                    parallelMergeSort(arr, left, middle, depth - 1);
                }

                #pragma omp section
                {
                    parallelMergeSort(arr, middle + 1, right, depth - 1);
                }
            }

            parallelMerge(arr, left, middle, right);
    }
}

void generate_random_arr(vector<int>& arr) {
	for (int i = 0; i < arr.size()	; i++) {
	arr[i] = rand()%100;
	}
}

int main() {
	int n = 0;
	printf("Enter number of elements in your array :  ");
    scanf("%d",&n);
   	vector<int> arr(n), arr_copy(n), arr_copy2(n);
   	generate_random_arr(arr);
	for(int i=0;i<n;i++){
		arr_copy[i]=arr[i];
		arr_copy2[i] = arr[i];
	}	
	
	auto start = chrono::high_resolution_clock::now();
	
	serialMergeSort(arr, 0, n - 1);
	
	auto end = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
	printf("Time taken for Serial Merge Sort of %d elements : %ld milliseconds\n", n, duration.count());

    // Parallel merge sort
    int depth = 8;  // Adjust this depth based on your system
    start = chrono::high_resolution_clock::now();
    
    parallelMergeSort(arr_copy, 0, n - 1, depth);
    
	end = chrono::high_resolution_clock::now();	
	duration = chrono::duration_cast<chrono::microseconds>(end - start);
	printf("Time taken for Parallel Merge Sort of %d elements : %ld milliseconds\n", n, duration.count());
	
    // Check if both sorts produced the same result
    int same = 1;
    for(int i=0;i<n;i++){
    	if(arr[i]!=arr_copy[i]){
    		same = 0;
    		break;
    	}
    }
	if(same == 0) printf("Parallel Merge Sort gave a different output from the Serial Merge Sort\n");
	printf("Enter 1 to see the initial and final arrays:  ");
	scanf("%d",&same);
	if(same){
		printf("Original Array : ");
		for(int i=0;i<n;i++) printf("%d ", arr_copy2[i]);
		printf("\n");
		printf("Sorted using Serial Merge Sort   : ");
		for(int i=0;i<n;i++) printf("%d ", arr[i]);
		printf("\n");
		printf("Sorted using Parallel Merge Sort : ");
		for(int i=0;i<n;i++) printf("%d ", arr_copy[i]);
		printf("\n");
	}	
	
    return 0;
}

