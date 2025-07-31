
#include "utils.c"
#include "omp.h"

// #include <scorep/SCOREP_User.h>

static int CUT_OFF_1;  // size limit for parallel MergeSort recursion
static int CUT_OFF_2;  // size limit for parallel Merge recursion

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

void insertionSort2(int* A, int l, int r, int* B, int s) {
    for(int i = l, k = s; i <= r; i++, k++)
        B[k] = A[i];

    for(int i = s+1; i <= s+r-l; i++) {
        int k = B[i], j = i-1;
        while (j >= s && B[j] > k) {
            B[j+1] = B[j];
            j = j-1;
        }
        B[j+1] = k;
    }
}

void merge(int* T, int l1, int r1, int l2, int r2, int* A, int l3) {
    int i = l1, j = l2, k = l3;

    while (i <= r1 && j <= r2)    
        if (T[i] <= T[j])
            A[k++] = T[i++];
        else
            A[k++] = T[j++];

    while (i <= r1)    
        A[k++] = T[i++];
    while (j <= r2)    
        A[k++] = T[j++];   
}

void merge2(int* A, int l, int m, int r, int* T) {
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

void mergeSort(int* A, int l, int r, int* B) {
    if (l < r) {
        int m = l+(r-l)/2;        // = (l+r)/2, but without overflow 
        mergeSort(A, l, m, B);
        mergeSort(A, m+1, r, B);
        merge2(A, l, m, r, B);
    }
}

int binarySearch(int k, int* T, int l, int r) {
    int i = l;
    int j = max(l, r+1);
    while (i < j) {
        int m = (i+j)/2;
        if (k <= T[m])
            j = m;
        else
            i = m+1;
    }
    return j;
}

void mergeP(int* T, int l1, int r1, int l2, int r2, int* A, int l3) {
    int n1 = r1-l1+1, 
        n2 = r2-l2+1;
    if (n1 < n2) {
        swap(&l1, &l2);
        swap(&r1, &r2);
        swap(&n1, &n2);
    }
    if (n1 == 0)                                  // implies n2 = 0
        return;
    else if (n1+n2 < CUT_OFF_2)
        merge(T, l1, r1, l2, r2, A, l3);           // fall back to serial merge   
    else {
        int m1 = l1+(r1-l1)/2;                     // = (l1+r1)/2, but without overflow
        int m2 = binarySearch(T[m1], T, l2, r2);
        int m3 = l3 + (m1-l1) + (m2-l2);
        A[m3] = T[m1];
        #pragma omp task
            mergeP(T, l1, m1-1, l2, m2-1, A, l3);
        // #pragma omp task                        // not necessary
            mergeP(T, m1+1, r1, m2, r2, A, m3+1);
        #pragma omp taskwait   
    }
}

void mergeSortPv2(int* A, int l, int r, int* B) {
    int n = r-l+1;
    if (n < CUT_OFF_1) {
        mergeSort(A, l, r, B);                   // fall back to serial merge sort
        // insertionSort(A, l, r);               // potentially even faster 
    } else if (l < r) {         
        int m = l+(r-l)/2;                       // = (l+r)/2, but without overflow
        #pragma omp task
            mergeSortPv2(B, l, m, A);
        // #pragma omp task                      // not necessary
            mergeSortPv2(B, m+1, r, A);
        #pragma omp taskwait    
            mergeP(A, l, m, m+1, r, B, l);
    }
}

void mergeSortPv2_(int* A, int l, int r, int* B, bool AtoB) {
    int n = r-l+1;
    if (n == 1) {
        if (AtoB)  
            B[l] = A[l];
    } else if (n < CUT_OFF_1) {
        mergeSort(A, l, r, B);                   // fall back to serial merge sort
        // insertionSort(A, l, r);               // potentially even faster
        if (AtoB)
            for (int i = l; i <= r; i++)    
            B[i] = A[i];
    } else if (l < r) {         
        int m = l+(r-l)/2;                       // = (l+r)/2, but without overflow
        #pragma omp task
            mergeSortPv2_(A, l, m, B, !AtoB);
        // #pragma omp task                      // not necessary
            mergeSortPv2_(A, m+1, r, B, !AtoB);
        #pragma omp taskwait    
            if (AtoB)    
            mergeP(A, l, m, m+1, r, B, l);
            else         
            mergeP(B, l, m, m+1, r, A, l); 
    }
}

void mergeSortPv2_CLRS(int* A, int l, int r, int* B, int s) {
    int n = r-l+1;
    if (n == 1)
        B[s] = A[l];
    else {         
        int m = l+(r-l)/2;                       // = (l+r)/2, but without overflow
        int m2 = m-l+1;
        int* T = malloc(n*sizeof(int));
        #pragma omp task
            mergeSortPv2_CLRS(A, l, m, T, 0);
        // #pragma omp task                      // not necessary
            mergeSortPv2_CLRS(A, m+1, r, T, m2);
        #pragma omp taskwait        
            mergeP(T, 0, m2-1, m2, n-1, B, s);
        free(T);   
    }
}

void sort(int* A, int n, int* B) {
    for (int i = 0; i < n; i++)    
        B[i] = A[i];
    #pragma omp parallel
    #pragma omp single
        mergeSortPv2(A, 0, n-1, B);  
}

void sort_(int* A, int n, int* B) {
    #pragma omp parallel
    #pragma omp single
        mergeSortPv2_(A, 0, n-1, B, true);  
}

void sort_CLRS(int* A, int n, int* B) {
    #pragma omp parallel
    #pragma omp single
        mergeSortPv2_CLRS(A, 0, n-1, B, 0);  
}

int main(int argc, char** argv) { 
    if (argc < 2)
        return -1;
    int   n        = atoi(argv[1]);  
    char* filePath = argc >= 3 ? argv[2] : NULL;  

    if(getenv("CUT_OFF_1"))
        CUT_OFF_1 = atoi(getenv("CUT_OFF_1"));
    else
        CUT_OFF_1 = 80;  
    if(getenv("CUT_OFF_2"))
        CUT_OFF_2 = atoi(getenv("CUT_OFF_2"));
    else
        CUT_OFF_2 = 8000;   
    dbg_print("OMP_NUM_THREADS = %s; omp_get_max_threads = %d; CUT_OFF_1 = %d; CUT_OFF_2 = %d \n", 
        getenv("OMP_NUM_THREADS"), omp_get_max_threads(), CUT_OFF_1, CUT_OFF_2);        

    //--------- Set up initial array
    int* A; 
    if (filePath != NULL) {
        dbg_print("Reading array of length %d from %s ...\n", n, filePath);   
        A = readArrayFromfile(filePath, n);
        if (A == NULL) {
        dbg_print("%s\n", "File not found, aborting ...");   
        exit(1);
        }
        //A = malloc(n*sizeof(int));
        //A[0] = 1, A[1] = 2, A[2] = 3, A[3] = 2, A[4] = 5, A[5] = 5, A[6] = 6; 
        //A[0] = 0, A[1] = 1, A[2] = 4, A[3] = 0, A[4] = 0; 
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
    int* B = malloc(n*sizeof(int));
    start = omp_get_wtime();
    sort(A, n, B);
    end   = omp_get_wtime();
    elapsed = end - start;

    //--------- Show results
    if (DBG_TEST) {
        if (isSorted(B, n))
            printf("Result is OK:\n");   
        else 
            printf("Result is BAD:\n");   
    }
    if (n < 100) {
        dbg_printArray(B, n);
        dbg_print("%s\n", "");
    }
    dbg_print("%s\n", "Elapsed time:");
    printf("%g\n", elapsed);

    //--------- Clean up
    free(A);   
    free(B);   
    return 0;   
}