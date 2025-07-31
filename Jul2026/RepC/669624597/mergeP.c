#include <string.h>
#include <time.h>

#include "omp.h"
#include "utils.c"

static int CUT_OFF;

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
   int n1 = r1 - l1 + 1, 
       n2 = r2 - l2 + 1;
   if (n1 < n2) {
     swap(&l1, &l2);
     swap(&r1, &r2);
     swap(&n1, &n2);
   }
   if (n1 == 0)  // implies n2 = 0
     return;
   else if (n1+n2 < CUT_OFF)
      merge(T, l1, r1, l2, r2, A, l3);   
   else {
      int m1 = (l1 + r1) / 2;
      int m2 = binarySearch(T[m1], T, l2, r2);
      int m3 = l3 + (m1 - l1) + (m2 - l2);
      A[m3] = T[m1];
      #pragma omp task
         mergeP(T, l1, m1-1, l2, m2-1, A, l3);
      #pragma omp task   
         mergeP(T, m1+1, r1, m2, r2, A, m3+1);
      #pragma omp taskwait   
   }
}

void insertionSort(int* A, int n) {
   for (int i = 1; i < n; i++) {
      int k = A[i];
      int j = i-1;
      while (j >= 0 && A[j] > k) {
         A[j + 1] = A[j];
         j = j-1;
      }
      A[j+1] = k;
   }
}

int* readOrCreate(int n, int n2, char* filePath) {
   int* A;
   if (filePath != NULL) {
      dbg_print("Reading sorted array of length %d from %s ...\n", n, filePath);   
      A = readArrayFromfile(filePath, n);
      if (A == NULL) {
         dbg_print("%s\n", "File not found, aborting ...");   
         exit(1);
      }
      if (!isSorted(A, n)) {
         printf("Array is not sorted, aborting ...\n");   
         exit(1);
      }
   } else { 
      dbg_print("Creating random sorted array of length %d ...\n", n);   
      A = malloc(n*sizeof(int));
      for (int i = 0; i < n; i++)		
         A[i] = rand() % (n+n2);
      insertionSort(A,n);   
   }
   return A;
}

int main(int argc, char** argv) { 
   if (argc < 3)
     return -1;
   int   n1        = atoi(argv[1]);  
   int   n2        = atoi(argv[2]);  
   char* filePath1 = argc >= 3 ? argv[3] : NULL;  
   char* filePath2 = argc >= 4 ? argv[4] : NULL;  

   //--------- set up initial arrays
   srand(time(NULL));
   int* A = readOrCreate(n1, n2, filePath1);
   int* B = readOrCreate(n2, n1, filePath2);
   if (n1+n2 < 100) {
     dbg_print("%s\n", "First array: ");   
     dbg_printArray(A, n1);
     dbg_print("%s\n", ""); 
     dbg_print("%s\n", "Second array: ");   
     dbg_printArray(B, n2);
     dbg_print("%s\n", ""); 
   }

   //--------- do timed merge
   dbg_print("%s\n", "Merging ...");
   double start, end, elapsed;
   int* C = malloc((n1+n1)*sizeof(int));
   int* T = malloc((n1+n1)*sizeof(int));
   int k = 0;
   for (int i = 0; i < n1; i++,k++)
      T[k] = A[i];
   for (int i = 0; i < n2; i++,k++)
      T[k] = B[i];   
   start = getWallTime();
   mergeP(T, 0, n1-1, n1, n1+n2-1, C, 0);
   end   = getWallTime();
   elapsed = end - start;

   //--------- show results
   if (DBG_TEST) {
      if (isSorted(C, n1+n2))
         printf("Result is OK:\n");   
      else 
         printf("Result is BAD:\n");   
   }
   if (n1+n2 < 100) { 
      dbg_printArray(C, n1+n2);
      dbg_print("%s\n", "");
   }
   dbg_print("%s\n", "Elapsed time:");
   printf("%g\n", elapsed);

   free(A);   
   free(B);
   free(C);   
   free(T);    
   return 0;   
}