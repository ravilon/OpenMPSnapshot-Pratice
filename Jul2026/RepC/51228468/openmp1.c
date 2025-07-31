#include <stdio.h>
#include <stdlib.h>
/***************************************************************/
#include <omp.h>
int main(void) {
    int np = omp_get_max_threads();
    if (np<2) exit(1);
    /* allocate from the heap with global visibility */
    int * A = malloc(np*sizeof(int));
    #pragma omp parallel shared(A)
    {
        int me = omp_get_thread_num();
        int B = 134;
        /* store local data B at PE 0 into A at PE 1 */
        if (me==0) A[1] = B;
        /* global synchronization of execution and data */
        #pragma omp barrier
        /* observe the result of the store */
        if (me==1) printf("A@1=%d\n",A[1]); fflush(stdout);
    }
    free(A);
    return 0;
}
/***************************************************************/
