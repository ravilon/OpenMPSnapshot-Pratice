#include "array.h"
#include "mpi.h"
#include "omp.h"
#include <math.h>

#define MASTER 0

void quick_sort_omp(double *A, int l, int r) {
  if (l < r) {
    int s = hoare_partition(A, l, r);

#pragma omp parallel sections
    {
#pragma omp section
      quick_sort_omp(A, l, s - 1);

#pragma omp section
      quick_sort_omp(A, s + 1, r);
    }
  }
}

void MPI_Qsort(double *A, int length, MPI_Comm comm) {

  int p = 0;
  MPI_Comm_size(comm, &p);

  // This will be the base case for the recursion
  if (p <= 1) {
    quick_sort_omp(A, 0, length - 1);
    return;
  }

  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  double *localA = NULL;

  // S is the array of elements smaller than the pivot
  // L is the array of elements larger than the pivot
  // count, displacement of S and L equivalent to prefix sum operations
  double *S = NULL;
  int lengthS = 0;
  double *localS = NULL;
  int localLengthS = 0;
  int countS[p];
  int displacementS[p];

  double *L = NULL;
  int lengthL = 0;
  double *localL = NULL;
  int localLengthL = 0;
  int countL[p];
  int displacementL[p];

  int count[p];
  int displacement[p];

  int pivot = 0;

  MPI_Comm commS;
  MPI_Comm commL;
  int pS = 0;
  int color = 0;

  // Consider a list of size n equally divided across p processors.
  vector_variant(p, length, count, displacement);

  memory_allocation(&localA, count[rank]);

  MPI_Scatterv(A, count, displacement, MPI_DOUBLE, localA, count[rank],
               MPI_DOUBLE, MASTER, comm);

  // A pivot is selected by one of the processors and made known to all
  // processors.
  if (rank == MASTER) {
    pivot = A[hoare_partition(A, 0, length - 1)];
  }

  MPI_Bcast(&pivot, 1, MPI_DOUBLE, MASTER, comm);

  split_array(       //
      localA,        //
      count[rank],   //
      pivot,         //
      &localS,       //
      &localLengthS, //
      &localL,       //
      &localLengthL  //
  );

  // Make every process know the countS and countL
  MPI_Allgather(&localLengthS, 1, MPI_INT, countS, 1, MPI_INT, comm);
  MPI_Allgather(&localLengthL, 1, MPI_INT, countL, 1, MPI_INT, comm);

  // Make every process know the displacementS and displacementL
  displacementS[0] = 0;
  displacementL[0] = 0;
  for (int i = 1; i < p; i++) {
    displacementS[i] = displacementS[i - 1] + countS[i - 1];
    displacementL[i] = displacementL[i - 1] + countL[i - 1];
  }

  // MASTER process will allocate memory for S and L
  if (rank == MASTER) {
#pragma omp parallel for
    for (int i = 0; i < p; i++) {
      lengthS += countS[i];
      lengthL += countL[i];
    }
    memory_allocation(&S, lengthS);
    memory_allocation(&L, lengthL);
  }

  MPI_Gatherv(       //
      localS,        //
      localLengthS,  //
      MPI_DOUBLE,    //
      S,             //
      countS,        //
      displacementS, //
      MPI_DOUBLE,    //
      MASTER,        //
      comm           //
  );
  MPI_Gatherv(       //
      localL,        //
      localLengthL,  //
      MPI_DOUBLE,    //
      L,             //
      countL,        //
      displacementL, //
      MPI_DOUBLE,    //
      MASTER,        //
      comm           //
  );

  if (rank == MASTER) {
    concat_array(S, lengthS, L, lengthL, A);
  }
  // The set of processors is partitioned into two.
  // In proportion of the size of lists L and U + 0.5.
  pS = (int)ceil(lengthS * p / length + 0.5);

  color = (rank < pS) ? 0 : 1; // assign colors to processes
  // Create new communicator commS  and commL
  MPI_Comm_split(comm, color, rank, &commS);
  MPI_Comm_split(comm, 1 - color, rank, &commL);

  if (lengthS > 1) {
    MPI_Qsort(A, lengthS, commS);
  }

  if (lengthL > 1) {
    MPI_Qsort(A + lengthS, lengthL, commL);
  }

  memory_deallocation(&localA);
  memory_deallocation(&localS);
  memory_deallocation(&localL);

  if (rank == MASTER) {
    memory_deallocation(&S);
    memory_deallocation(&L);
  }

  MPI_Comm_free(&commS);
  MPI_Comm_free(&commL);
}