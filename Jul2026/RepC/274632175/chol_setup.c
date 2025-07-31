#include "chol_setup.h"

#include <stdlib.h>
#include <time.h>

#include "fptype.h"
#include "genmat.h"
#include "densutil.h"
#include "matfprint.h"


extern fp_t *Aorig;
extern fp_t *A;


int chol_setup(int check, int m, int mr, int ts, int bs, int tm, int mleft) 
{
// Creation of data matrices
#if USE_LL || USE_RL
A = GENMAT_SPD(m, m);
#pragma omp register ([m*m]A)
#else
#warning "Zhuang, this needs to be fixed to work with matrices that are not an integral number of blocks"
A = calloc( mr * mr, sizeof(fp_t));
#endif
if ( A == NULL ) return 1;

//	Aorig = malloc(m*m*sizeof(fp_t));
if ( check ) {
#if USE_LL || USE_RL
Aorig = DMAT_CP(m, m, A, m);
#else 
Aorig = DMAT_CP(mr, mr, A, mr);
#endif
}

return 0;
}


void chol_shutdown() 
{
free(A);
free(Aorig);
}
