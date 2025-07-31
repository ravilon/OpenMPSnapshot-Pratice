#include <iostream>
#include <ctime>
#include <omp.h>
#include <cmath>
#include <zconf.h>
#include "sort.h"

using namespace std;

const int SIZE = 10;

void print_array(int arr[]) {
    for (int i = 0; i < SIZE; i++) {
        cout << arr[i] << " ";
    }
}

int main() {

    int size, rank;
    int k = 5;

//#pragma omp parallel private(k, rank) num_threads(10)
//    {
//        k = 5;
//        size = omp_get_num_threads();
//        rank = omp_get_thread_num();
//
//        if (!rank) {
//            k++;
//        }
//
//        printf("Hello from %d - k = %d out of %d\n", rank, k, size);
//    }

//#pragma omp parallel sections
//    {
//#pragma omp section
//        {
//            rank = omp_get_thread_num();
//            printf("section 1 thread %d\n", rank);
//            sleep(100);
//        }
//#pragma omp section
//        {
//            rank = omp_get_thread_num();
//            printf("section 2 thread %d\n", rank);
//        }
//#pragma omp section
//        {
//            rank = omp_get_thread_num();
//            printf("section 3 thread %d\n", rank);
//        }
//#pragma omp section
//        {
//            rank = omp_get_thread_num();
//            printf("section 4 thread %d\n", rank);
//        }
//    }

    int sum;
#pragma omp parallel for reduction(+:sum) //schedule(guided, 10) num_threads(4)
    for (int i = 0; i < 100; i++) {
        rank = omp_get_thread_num();
        printf("iteration - %d, thread - %d\n", i, rank);
    }

    cout << "last - " << rank << endl;
}

