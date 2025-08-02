#pragma once

// void swap(int num1, int num2) {
// 	int temp = intArray[num1];
// 	intArray[num1] = intArray[num2];
// 	intArray[num2] = temp;
// }

void swap(int *x, int *y)
{
	int temp = *x;
	*x = *y;
	*y = temp;
}

// int partition(int left, int right, int pivot) {
// 	int leftPointer = left - 1;
// 	int rightPointer = right;
//
// 	while (true) {
// 		while (intArray[++leftPointer] < pivot) {
// 			//do nothing
// 		}
//
// 		while (rightPointer > 0 && intArray[--rightPointer] > pivot) {
// 			//do nothing
// 		}
//
// 		if (leftPointer >= rightPointer) {
// 			break;
// 		}
// 		else {
// 			printf(" item swapped :%d,%d\n", intArray[leftPointer], intArray[rightPointer]);
// 			swap(leftPointer, rightPointer);
// 		}
// 	}
//
// 	printf(" pivot swapped :%d,%d\n", intArray[leftPointer], intArray[right]);
// 	swap(leftPointer, right);
// 	printf("Updated Array: ");
// 	display();
// 	return leftPointer;
// }
//
// inline void quicksort_serial(int arr[], int left, int right) {
// 	if (right - left <= 0) {
// 		return;
// 	}
// 	int pivot = arr[right];
// 	int partitionPoint = partition(left, right, pivot);
// 	quicksort_serial(arr, left, partitionPoint - 1);
// 	quicksort_serial(arr, partitionPoint + 1, right);
//
// }

inline int choose_pivot(int i, int j)
{
	return((i + j) / 2);
}

inline void quicksort_serial(int arr[], int left, int right)
{
	int key, i, j, pivot;

	if (left < right)
	{
		pivot = choose_pivot(left, right); //may be random instead of the middle
		swap(&arr[left], &arr[pivot]);
		key = arr[left];
		i = left + 1;
		j = right;
		while (i <= j)
		{
			while ((i <= right) && (arr[i] <= key))
				i++;
			while ((j >= left) && (arr[j] > key))
				j--;
			if (i < j)
				swap(&arr[i], &arr[j]);
		}
		/* swap two elements */
		swap(&arr[left], &arr[j]);

		/* the smaller array */
		quicksort_serial(arr, left, j - 1);
		quicksort_serial(arr, j + 1, right);
	}
}