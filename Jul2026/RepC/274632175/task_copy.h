#ifndef __TASK_COPY_H__
#define __TASK_COPY_H__


#include "selfsched.h"
#include "async_struct.h"


#define LIBBBLAS_EXPORT __attribute__((__visibility__("default")))

#define OMPSS_PRIOR_DFLT 1

// double precision
#pragma omp task in([bm]X) out([bm]Y) priority(p) label(dcopy) no_copy_deps
extern LIBBBLAS_EXPORT void task_dcopy(int p, int bm, int bn, int m, int n, double *X, double *Y);  

//#pragma omp task in([bm]X) out([bm]Y) priority(p)
//extern LIBBBLAS_EXPORT void task_dcopy_sched(async_t *sync, int z, int p, int bm, int bn, int m, int n, double *X, int iy, selfsched_t *schedY, double *Y);  


// single precision
#pragma omp task in([bm]X) out([bm]Y) priority(p) label(scopy) no_copy_deps
extern LIBBBLAS_EXPORT void task_scopy(int p, int bm, int bn, int m, int n, float *X, float *Y);  

//#pragma omp task in([bm]X) out([bm]Y) priority(p)
//extern LIBBBLAS_EXPORT void task_scopy_sched(async_t *sync, int z, int p, int bm, int bn, int m, int n, float *X, int iy, selfsched_t *schedY, float *Y);  


#endif // __TASK_COPY_H__
