#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "mkl.h"
#include "mkl_spblas.h"

#include "cg_main.h"
#include "vector.h"
#include "sparsematrix.h"
#include "finout.h"
#include "cg_kernels.h"
#include "cg_kernels_lookahead.h"


// rhs = right-hand side
// x   = computed solution
// res = residual
// p   = conjugate direction
// b   = block size
long cg(SparseMatrix *A, double *rhs, double *x, int s, int b, int maxit) {
	int n = A->dim1; 
	double *mat = A->vval;
	int *vptr = A->vptr;
	int *vpos = A->vpos;
	int nb = (n + b - 1) / b;
	int l = n - (nb-1) * b;
	int bfin = (b==n)? 0 : l;
	int ifin = (b==n)? 0 : n - l;

	double *z = (double*) malloc(sizeof(double) * n);
	if ( z == NULL ) {
		fprintf(stderr, "error: could not allocate z\n");
		return 1;
	}
	double *ztmp = (double*) calloc(n * nb, sizeof(double));
	if ( ztmp == NULL ) { 
		fprintf(stderr, "error: could not allocate ytmp\n");
		return 1;
	}
	double *res = (double*) malloc(sizeof(double) * n);
	if ( res == NULL ) {
		fprintf(stderr, "error: could not allocate res\n");
		return 1;
	}
	double *p = (double*) malloc(sizeof(double) * n);
	if ( p == NULL ) {
		fprintf(stderr, "error: could not allocate p\n");
		return 1;
	}


	int IONE = 1; 
	double DONE = 1.0, DMONE = -1.0;

	struct timeval start;
	gettimeofday(&start,NULL);

	mkl_dcsrsymv ("U", &n, mat, vptr, vpos, x, z);  		// z = A * x
	dcopy (&n, rhs, &IONE, res, &IONE);                            	// res = b
	daxpy (&n, &DMONE, z, &IONE, res, &IONE);                      	// res -= z
	dcopy (&n, res, &IONE, p, &IONE);                              	// p = res
	double gamma = ddot (&n, res, &IONE, res, &IONE);              	// gamma = res' * res
	double zz;
	double zp;
	double tol = sqrt (gamma);                                     	// tol = sqrt(gamma) = norm (res)
	double umbral = 1.0e-8;
	double alpha=1.0;
	double beta;
	int iter = 0;
	while ((iter < maxit) && (tol > umbral)) {
		int i; 

		// z = A * p
		for ( i = 0; i < n; i+=b ) {
			task_mv(i, b, n, nb, A, p, &z[i], ztmp);
		}
		for ( i = 0; i < n; i+=b ) {
			task_zred(i, b, n, nb, ztmp, &z[i]);
		}

		// zp = z' * p
		// zz = z' * z
		lhtask_ddot_init(b, p, z, &zp, &zz); 
		for ( i=b; i<n-b; i+=b ) {
			lhtask_ddot(b, &p[i], &z[i], &zp, &zz); 
		}
		// alpha = gamma / zp
		// beta = gamma * zz / zp^2 - 1
		// gamma = gamma * beta
		lhtask_ddot_fin(bfin, &p[ifin], &z[ifin], &zp, &zz, &alpha, &beta, &gamma); 

// --- dot product for alpha : blocks of z and p are loaded from main memory

		// x += alpha * p;
		// res -= alpha * z
		// p = res + beta * p
		for ( i=0; i<n; i+=b ) {
			lhtask_axpy(i, b, n, &alpha, &beta, &p[i], &z[i], &x[i], &res[i]); 
		}

		iter++;

		#pragma omp taskwait on(gamma)
		#pragma omp taskwait 

		tol = sqrt (gamma);      

#if 0
		FILE *fstrm = fopen("lookahead.log", "a");
		if ( fstrm == NULL ) {
			fprintf(stderr, "warning: failed to open %s\n", fstrm);
		}
#endif
		printf ("iter %i : alpha %16.20e beta %e %16.20e gamma %e tol %16.20e\n",iter, alpha, beta, beta, gamma, tol);
#if 0
		fprintf (fstrm, "iter %i : alpha %e beta %e %.20f gamma %e tol %e\n",iter, alpha, beta, beta, gamma, tol);
		fclose(fstrm);
		print_dvector ("lookahead.log", "x", n, x); 
		print_dvector ("lookahead.log", "p", n, p); 
#endif
	}

	//print_dvector("x.la", "x", n, x);

	struct timeval stop;
	gettimeofday(&stop,NULL);

	//printf ("it %i res %e\n", iter, tol);

	free(ztmp);
	free(res);
	free(p);

	unsigned long elapsed = (stop.tv_sec - start.tv_sec) * 1000000;
	elapsed += (stop.tv_usec - start.tv_usec);

	return elapsed;
} 
