#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_SIZE 50
void swap(int* a, int* b)
{
    int t = *a;
    *a = *b;
    *b = t;
}

int partition(int arr[], int low, int high)
{
    int pivot = arr[high];
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}
void quickSort(int arr[], int low, int high) //recursive decomposition
{
    if (low < high) {
        int pi = partition(arr, low, high);

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                quickSort(arr, low, pi - 1);
            }

            #pragma omp section
            {
                quickSort(arr, pi + 1, high);
            }
        }
    }
}



int main()
{
    int m, n;
    int matrix[MAX_SIZE][MAX_SIZE];

    printf("Enter the number of rows (m): ");
    scanf("%d", &m);
    printf("Enter the number of columns (n): ");
    scanf("%d", &n);
    printf("Enter the elements of the matrix:\n");


    for (int i = 0; i < m; i++) 
    {
        for (int j = 0; j < n; j++) {
            scanf("%d", &matrix[i][j]);
        }
    }

    int lastDigit = 7; 
     if (lastDigit == 7) 
    {
        #pragma omp parallel for num_threads(m)
        for (int i = 0; i < m; i++) {
            quickSort(matrix[i], 0, n - 1);
        }
    }
    

    int matrixSum = 0;
    printf("Sorted Matrix:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", matrix[i][j]);
            matrixSum += matrix[i][j];
        }
        printf("\n");
    }

    printf("Matrix Sum: %d\n", matrixSum);

    return 0;
}
