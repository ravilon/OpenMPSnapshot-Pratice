#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "cgpipe_main.h"

#include "fptype.h"
#include "fpblas.h"
#include "task_log.h"
#include "matfprint.h"

#include "cgpipe_workspace.h"
#include "as_man.h"
#include "async_struct.h"

#include "bblas_dot.h"
#include "bblas_dot_async.h" 
#include "bblas_gemm.h"
#include "bblas_axpy.h" 
#include "bblas_axpy_async.h" 
#include "bblas_copy.h"


typedef enum {
DOT_ALPHA1, 
DOT_ALPHA2,
DOT_ALPHA3, 
} dot_t;


/* The first CP preferably finishes before the first GEMM executes, so that we have a good profile of p. The asynchronous DOT
should be preferred over the GEMM, because we want to overlap the DOTs with the GEMMs, and complete as much of the dot product as possible. 
The subsequent AXPY must have lowest priority, because we only want to use an imprecise dot product if it improves performance. */
#define PRIOR_FIRSTCP		9999999
#define PRIOR_GEMM			99999
#define PRIOR_ASYNCDOT		999999
#define PRIOR_ASYNCAXPY		9999


/* Algorithm 2 in Ghysels, Vanroose, 2012 */
int CGPIPE(int bm, int bn, int n, void *A, int s, fp_t *b, fp_t *xbuff, int *offs, fp_t tol, int steps, fp_t *work, unsigned long works, int lookahead, int async, fp_t profile, int warmup, int cglog) 
{
/* initializations */
fp_t *x[4] = {xbuff, xbuff+n, xbuff+2*n, xbuff+3*n};
fp_t **r; 
fp_t *w;  
fp_t *q;  
fp_t *p; 
fp_t *z; 
fp_t *ss; 
fp_t **alpha;
fp_t **gamma;
fp_t **delta;

/* allocate work space */
int dupl = 4;
if ( cgpipe_malloc(n, s, dupl, work, works, &p, &w, &q, &z, &ss, &r, &alpha, &gamma, &delta) ) {
return -1;
}

/* initializations */
async_t *sync =  async_init(work, 3, n, bm, -1, 1);
asman_t *asman = asman_init(4);
if ( cglog ) {
async_log(&sync[DOT_ALPHA1], 256, "cgsched");
async_log(&sync[DOT_ALPHA2], 1024, "cgsched");
async_log(&sync[DOT_ALPHA3], 1024, "cgsched");
}
(gamma[0])[0] = FP_MONE;

/* r[0] = b - A * x[0] */
BBLAS_COPY(PRIOR_FIRSTCP, bm, bn, n, s, b, r[0]);	                     						
#if USE_SPARSE
BSBLAS_CSRMMB(bm, s, FP_MONE, A, r[0], n, FP_ONE, r[0], n);
#else
BBLAS_GEMM(PRIOR_FIRSTCP, OMPSSBLAS_NTRANSP, OMPSSBLAS_NTRANSP, bm, bn, bm, n, s, n, FP_MONE, A, n, x[0], n, FP_ONE, r[0], n); 					
#endif

/* w[0] = A * r[0]*/
#if USE_SPARSE
BSBLAS_CSRMMB(bm, s, FP_ONE, A, r[0], n, FP_NOUGHT, w[0], n);
#else
BBLAS_GEMM(PRIOR_FIRSTCP, OMPSSBLAS_NTRANSP, OMPSSBLAS_NTRANSP, bm, bn, bm, n, s, n, FP_ONE, A, n, r[0], n, FP_NOUGHT, w, n); 					
#endif

int iprev = 0;
async_stat_t status = STAT_AHEAD;

int i, k; 
for ( k=0; k<steps; ++k ) {
i = asman_nexti(asman);

BBLAS_DOT4(k, &sync[DOT_ALPHA1], bm, bn, n, s, gamma[iprev], alpha[iprev], r[iprev], w, gamma[i], delta[i], alpha[i]);

/* q = A * w */
#if USE_SPARSE
status = BSBLAS_CSRMMB_PRED(bm, s, FP_ONE, A, w, n, FP_NOUGHT, q, n,\
k, i, &sync[DOT_ALPHA1], gamma, tol);
#else
status  = BBLAS_GEMM_PRED(bm, bn, bm, n, s, n, FP_ONE, A, w, FP_NOUGHT, q, k, i, asman, &sync[DOT_ALPHA1], gamma, tol);
#endif
if ( status == STAT_AHEAD ) {
#pragma omp taskwait on (*gamma[i])
ASMAN_BREAK(k, i, asman, gamma, &sync[DOT_ALPHA1], 1, tol);
}
if ( status == STAT_CONVERGED ) {
break;
}

BBLAS_AXPY3(bm, bn, n, s, gamma[i], gamma[iprev], alpha[i], z, ss, p, q, w, x[iprev], x[i], r[iprev], r[i]);

iprev = i;
status = STAT_AHEAD;
}

#pragma omp taskwait

*offs = iprev;

asman_fini(asman);

free(r);

return k;
} 
