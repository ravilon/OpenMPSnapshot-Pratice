#include <math.h>
#include <mpi.h>
#include "omp.h"

#include "utils.c"

static int WORLD_RANK;
static int WORLD_SIZE;
static int WORLD_MAX_RANK;
static int CUT_OFF_1;      // size limit for parallel multithread MergeSort recursion
static int CUT_OFF_2;      // size limit for parallel multithread Merge recursion

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

void merge2(int* T, int l1, int r1, int l2, int r2, int* A, int l3) {
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
    if (n1 == 0)                                   // implies n2 = 0
        return;
    else if (n1+n2 < CUT_OFF_2)
        merge2(T, l1, r1, l2, r2, A, l3);          // fall back to serial merge   
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

void mergeSort(int* A, int l, int r, int* T) {
    if (l < r) {
        int m = l+(r-l)/2;  // = (l+r)/2, but without overflow 
        mergeSort(A, l, m, T);
        mergeSort(A, m+1, r, T);
        merge(A, l, m, r, T);
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

int topmostLevel(int rank) {
    int level = 0;
    while (pow(2, level) <= rank)
        level++;
    return level;
}

void sortParallel(int* A, int n, int* T, int level, MPI_Comm comm) {
    int helper_rank = WORLD_RANK + pow(2, level);
    if (helper_rank > WORLD_MAX_RANK) {   // if no more processes available 
        dbg_print("No more processes available, on rank %d\n", WORLD_RANK);
        //--------- Sort in parallel by multithreading
        for(int i = 0; i < n; i++)
            T[i] = A[i];
        #pragma omp parallel
        #pragma omp single
            mergeSortPv2(T, 0, n-1, A);
    } else {
        dbg_print("Process %d has helper %d\n", WORLD_RANK, helper_rank);
        MPI_Request request;
        MPI_Status status;
        const int m = n/2;
        //--------- Send second half, asynchronous
        MPI_Isend(A+m, n-m, MPI_INT, helper_rank, 0, comm, &request);
        //--------- Sort first half
        sortParallel(A, m, T, level+1, comm);       
        //--------- Receive second half sorted
        MPI_Request_free(&request);
        MPI_Recv(A+m, n-m, MPI_INT, helper_rank, 0, comm, &status);
        //--------- Merge the two sorted sub-arrays in parallel by multithreading
        for(int i = 0; i < n; i++)
            T[i] = A[i];
        #pragma omp parallel
        #pragma omp single       
            mergeP(T, 0, m-1, m, n-1, A, 0);
    }
}

void sortHelper(MPI_Comm comm) {
    int level = topmostLevel(WORLD_RANK);
    dbg_print("Helper process %d on level %d\n", WORLD_RANK, level);
    //--------- Probe for a message and determine its size and sender
    MPI_Status status;
    int size;
    MPI_Probe(MPI_ANY_SOURCE, 0, comm, &status);
    MPI_Get_count(&status, MPI_INT, &size);
    int parent_rank = status.MPI_SOURCE;  
    //--------- Receive sub-array and sort
    int* chunk = malloc(size*sizeof(int));
    int* tempArray = malloc(size*sizeof(int));
    MPI_Recv(chunk, size, MPI_INT, parent_rank, 0, comm, &status);
    sortParallel(chunk, size, tempArray, level, comm);
    //--------- Send sorted array to parent process
    MPI_Send(chunk, size, MPI_INT, parent_rank, 0, comm);
    free(chunk);   
    free(tempArray);  
}

void sort(int* A, int n, int* T) {
    if (WORLD_RANK == 0)
        sortParallel(A, n, T, 0, MPI_COMM_WORLD);
    else
        sortHelper(MPI_COMM_WORLD);
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

    //--------- Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &WORLD_RANK);
    MPI_Comm_size(MPI_COMM_WORLD, &WORLD_SIZE);
    WORLD_MAX_RANK = WORLD_SIZE - 1;

    //--------- Set up initial array at root
    int* A; 
    if (WORLD_RANK == 0) {
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
    }

    //--------- Do timed sorting
    int* T;
    if (WORLD_RANK == 0) {
        dbg_print("%s\n", "Sorting ...");
        T = malloc(n*sizeof(int)); 
    }
    double start, end, elapsed;
    MPI_Barrier(MPI_COMM_WORLD); 
    start = MPI_Wtime();
    sort(A, n, T);
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    elapsed = end - start;

    //--------- Show results at root
    if (WORLD_RANK == 0) {
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
    }

    //--------- Clean up and Finalize MPI
    if (WORLD_RANK == 0) {
        free(A);   
        free(T);  
    }
    MPI_Finalize();  
    return 0;   
}