#include <stdio.h>
#include <stdlib.h>
#define TRUE  1
#define FALSE 0
#include <omp.h>

int main(int argc, char *argv[]) {
int *a;
int n = 2; 
int nthreads, vlen, indx, offset = 4, i, TID;
int failed;
(void) omp_set_num_threads(3);
indx = offset;

/*  Prepara os parâmetros para a computação e aloca memória */
#pragma omp parallel firstprivate(indx) shared(a,n,nthreads,failed)
{
#pragma omp single

nthreads = omp_get_num_threads();
vlen = indx + n*nthreads;
if ( (a = (int *) malloc(vlen*sizeof(int))) == NULL )
failed = TRUE;
else
failed = FALSE;
} /*-- End of parallel region --*/
for(i=0; i<vlen; i++) a[i] = -i-1;
/* Cada thread acessa o vetor com a variável indx */  
printf("Comprimento do segmento por thread é %d\n", n);
printf("O offset do vetor a é %d\n",indx);
#pragma omp parallel default(none) firstprivate(indx) private(i,TID) shared(n,a)
{
TID = omp_get_thread_num();
indx += n*TID;
for(i=indx; i<indx+n; i++)
a[i] = TID + 1;
} /*-- Final da região paralela --*/
printf("Depois da região paralela:\n");
for (i=0; i<vlen; i++)
printf("a[%d] = %d\n",i,a[i]);
free(a);
return(0);
}
