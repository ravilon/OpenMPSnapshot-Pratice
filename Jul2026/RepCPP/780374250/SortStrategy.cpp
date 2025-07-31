#include <algorithm>
#include <omp.h>

template <typename T>
class SortStrategy {
public:
    void virtual reset(T* arr, T* original_array, int size) { }

    void virtual bubbleSort(T* arr, int size) {}

    void virtual oddEvenBubbleSort(T* arr, int size) {}

    void virtual shellSort(T* arr, int size) {}

    void virtual quickSort(T* arr, int size) {}
};

template <typename T>
class SequenceSortStrategy: public SortStrategy<T> {
public:

    void reset(T* arr, T* original_array, int size) override {
        for (int i = 0; i < size; ++i) {
            arr[i] = original_array[i];
        }
    }

    void bubbleSort(T* arr, int size) override {
        for (int i = 0; i < size - 1; ++i) {
            for (int j = 0; j < size - i - 1; ++j) {
                if (arr[j] > arr[j + 1]) {
                    std::swap(arr[j], arr[j + 1]);
                }
            }
        }
    }

    void oddEvenBubbleSort(T* arr, int size) override {
        bool sorted = false;
        while (!sorted) {
            sorted = true;
            for (int i = 1; i < size - 1; i += 2) {
                if (arr[i] > arr[i + 1]) {
                    std::swap(arr[i], arr[i + 1]);
                    sorted = false;
                }
            }
            for (int i = 0; i < size - 1; i += 2) {
                if (arr[i] > arr[i + 1]) {
                    std::swap(arr[i], arr[i + 1]);
                    sorted = false;
                }
            }
        }
    }

    void shellSort(T* arr, int size) override {
        for (int gap = size / 2; gap > 0; gap /= 2) {
            for (int i = gap; i < size; ++i) {
                int temp = arr[i];
                int j;
                for (j = i; j >= gap && arr[j - gap] > temp; j -= gap) {
                    arr[j] = arr[j - gap];
                }
                arr[j] = temp;
            }
        }
    }

    void quickSort(T* arr, int size) override {
        quickSort(arr, 0, size - 1);
    }

private:
    void quickSort(T* arr, int low, int high) {
        if (low < high) {
            int pivotIndex = partition(arr, low, high);
            quickSort(arr, low, pivotIndex - 1);
            quickSort(arr, pivotIndex + 1, high);
        }
    }

    int partition(T* arr, int low, int high) {
        int pivot = arr[high];
        int i = low - 1;

        for (int j = low; j <= high - 1; ++j) {
            if (arr[j] <= pivot) {
                ++i;
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }

        int temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;

        return i + 1;
    }

};

template <typename T>
class ParallelSortStrategy : public SortStrategy<T> {
public:
    void reset(T* arr, T* original_array, int size) override {
        int i;
        #pragma omp parallel for private(i)
        for (i = 0; i < size; ++i) {
            arr[i] = original_array[i];
        }
    }

    void bubbleSort(T* arr, int size) override {
        #pragma omp parallel
        {
            int i, j;
            #pragma omp for collapse(2) private(i, j)
            for (i = 0; i < size - 1; ++i) {
                for (int j = 0; j < size - i - 1; ++j) {
                    if (arr[j] > arr[j + 1]) {
                        std::swap(arr[j], arr[j + 1]);
                    }
                }
            }
        }
    }

    void oddEvenBubbleSort(T* arr, int size) override {
        bool sorted = false;
        while (!sorted) {
            int i;
            sorted = true;
            #pragma omp parallel sections private(i)
            {
                #pragma omp section
                for (i = 1; i < size - 1; i += 2) {
                    if (arr[i] > arr[i + 1]) {
                        std::swap(arr[i], arr[i + 1]);
                        sorted = false;
                    }
                }

                #pragma omp section
                for (i = 0; i < size - 1; i += 2) {
                    if (arr[i] > arr[i + 1]) {
                        std::swap(arr[i], arr[i + 1]);
                        sorted = false;
                    }
                }
            }
        }
    }

    void shellSort(T* arr, int size) override {
        #pragma omp parallel
        {
            int i;
            for (int gap = size / 2; gap > 0; gap /= 2) {
                #pragma omp for private(i)
                for (i = gap; i < size; ++i) {
                    int temp = arr[i];
                    int j;
                    for (j = i; j >= gap && arr[j - gap] > temp; j -= gap) {
                        arr[j] = arr[j - gap];
                    }
                    arr[j] = temp;
                }
            }
        }
    }

    void quickSort(T* arr, int size) override {
        #pragma omp parallel
        {
            #pragma omp single nowait
            {
                quickSort(arr, 0, size - 1);
            }
        }
    }

private:
    void quickSort(T* arr, int low, int high) {
        if (low < high) {
            int pivotIndex = partition(arr, low, high);
            #pragma omp parallel sections
            {
                #pragma omp section
                {
                    quickSort(arr, low, pivotIndex - 1);

                }
                #pragma omp section
                {
                    quickSort(arr, pivotIndex + 1, high);
                }
            }
        }
    }

    int partition(T* arr, int low, int high) {
        int pivot = arr[high];
        int i = low - 1;

        int j;
        for (j = low; j <= high - 1; ++j) {
            if (arr[j] <= pivot) {
                {
                    ++i;
                    int temp = arr[i];
                    arr[i] = arr[j];
                    arr[j] = temp;
                }
            }
        }

        int temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;

        return i + 1;
    }


};