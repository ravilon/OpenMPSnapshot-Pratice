
#include "omp.h"
#include "utils.c"

// #include <scorep/SCOREP_User.h>

static int CUT_OFF;  // size limit for parallel MergeSort recursion

void insertionSort(int* A, int l, int r) {
    for (int i = l+1; i <= r; i++) {
        int k = A[i], j = i-1;
        while (j >= l && A[j] > k) {
            A[j+1] = A[j];
            j = j-1;
        }
        A[j+1] = k;
    }
}

void merge(int* A, int l, int m, int r, int* T) {
    int i = l, j = m+1, k = l;

    while (i <= m && j <= r)    
        if (A[i] <= A[j])
            T[k++] = A[i++];
        else
            T[k++] = A[j++];

    while (i <= m)    
        T[k++] = A[i++];
    while (j <= r)    
        T[k++] = A[j++];   

    for (k = l; k <= r; k++)
        A[k] = T[k];
}

void mergeSort(int* A, int l, int r, int* T) {
    if (l < r) {
        int m = l+(r-l)/2;  // = (l+r)/2, but without overflow 
        mergeSort(A, l, m, T);
        mergeSort(A, m+1, r, T);
        merge(A, l, m, r, T);
    }
}

void mergeSortPv1(int* A, int l, int r, int* T) {
    int n = r-l+1;
    if (n < CUT_OFF)
        mergeSort(A, l, r, T);          // fall back to serial merge sort
        // insertionSort(A, l, r);      // potentially even faster
    else if (l < r) {   
        int m = l+(r-l)/2;              // = (l+r)/2, but without overflow   
        #pragma omp task
            mergeSortPv1(A, l, m, T);
        // #pragma omp task             // not necessary
            mergeSortPv1(A, m+1, r, T);
        #pragma omp taskwait   
            merge(A, l, m, r, T);
    }
}

void mergeSortPv1_CLRS(int* A, int l, int r) {
    int n = r-l+1;
    if (l < r) {   
        int m = l+(r-l)/2;                // = (l+r)/2, but without overflow   
        int* T = malloc(n*sizeof(int));
        #pragma omp task
            mergeSortPv1_CLRS(A, l, m);
        // #pragma omp task               // not necessary
            mergeSortPv1_CLRS(A, m+1, r);
        #pragma omp taskwait   
            merge(A, l, m, r, T);
        free(T);
    }
}

void sort(int* A, int n, int* T) {
    #pragma omp parallel
    #pragma omp single
        mergeSortPv1(A, 0, n-1, T);
}

void sort_CLRS(int* A, int n) {
    #pragma omp parallel
    #pragma omp single
        mergeSortPv1_CLRS(A, 0, n-1);
}

int main(int argc, char** argv) { 
    if (argc < 2)
        return -1;
    int   n        = atoi(argv[1]);  
    char* filePath = argc >= 3 ? argv[2] : NULL;  

    if(getenv("CUT_OFF"))
        CUT_OFF = atoi(getenv("CUT_OFF"));
    else
        CUT_OFF = 80;  
    dbg_print("OMP_NUM_THREADS = %s; omp_get_max_threads = %d; CUT_OFF = %d \n", 
        getenv("OMP_NUM_THREADS"), omp_get_max_threads(), CUT_OFF);     

    //--------- Set up initial array
    int* A; 
    if (filePath != NULL) {
        dbg_print("Reading array of length %d from %s ...\n", n, filePath);   
        A = readArrayFromfile(filePath, n);
        if (A == NULL) {
            dbg_print("%s\n", "File not found, aborting ...");   
            exit(1);
        }
    } else { 
        dbg_print("Creating random array of length %d ...\n", n);   
        A = malloc(n*sizeof(int));
        srand(time(NULL));
        for (int i = 0; i < n; i++)		
            A[i] = rand() % n;
    }
    if (n < 100) {
        dbg_print("%s\n", "Initial array: ");   
        dbg_printArray(A, n);
        dbg_print("%s\n", ""); 
    }

    //--------- Do timed sorting
    dbg_print("%s\n", "Sorting ...");
    double start, end, elapsed;
    int *T = malloc(n*sizeof(int));
    start = omp_get_wtime();         // comment timings 
    // #pragma omp parallel          // and parallelize here to run scorep
    // #pragma omp single
        sort(A, n, T);
    end   = omp_get_wtime();
    elapsed = end - start;

    //--------- Show results
    if (DBG_TEST) {
        if (isSorted(A, n))
            printf("Result is OK:\n");   
        else 
            printf("Result is BAD:\n");   
    }   
    if (n < 100) {
        dbg_printArray(A, n);
        dbg_print("%s\n", "");
    }
    dbg_print("%s\n", "Elapsed time:");
    printf("%g\n", elapsed);

    //--------- Clean up
    free(T); 
    free(A);   
    return 0;   
}