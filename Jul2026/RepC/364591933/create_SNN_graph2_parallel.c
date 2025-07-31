#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void create_SNN_graph2_prallel (int N, int *row_ptr, int *col_idx, int **SNN_val, int threads){

int len = row_ptr[N];

*SNN_val = (int*)calloc(len,sizeof(int));



#pragma omp parallel for num_threads(threads)
for (int i = 1; i < N+1; ++i)
{
for (int j = row_ptr[i-1]; j < row_ptr[i]; ++j)
{
for (int k = row_ptr[col_idx[j]]; k < row_ptr[col_idx[j]+1]; ++k)
{
for (int l = row_ptr[i-1]; l < row_ptr[i]; ++l){
if (col_idx[l] == col_idx[k]) (*SNN_val)[j]+=1;
}
}
}
}

}