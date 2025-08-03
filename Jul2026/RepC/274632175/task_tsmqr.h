#ifndef __TASK_TSMQR_H__
#define __TASK_TSMQR_H__


#define LIBBBLAS_EXPORT __attribute__((__visibility__("default")))


#ifdef SINGLE_PRECISION
#define TASK_TSMQR 			task_stsmqr
#else
#define TASK_TSMQR 			task_dtsmqr
#endif




#pragma omp task in( [ibs*its1]Sh ) inout( [its1*its2]Fh, [its1*its2]Gh) priority(p)
extern LIBBBLAS_EXPORT void task_stsmqr( int ibs, int its1, int its2, int skip, float * Dh, /*int ldim_D,*/float * Sh, /*int ldim_S,*/float * Fh, /*int ldim_F,*/float * Gh/*, int ldim_G */, int p);

#pragma omp task in( [ibs*its1]Sh ) inout( [its1*its2]Fh, [its1*its2]Gh) priority(p)
extern LIBBBLAS_EXPORT void task_dtsmqr( int ibs, int its1, int its2, int skip, double * Dh, /*int ldim_D,*/double * Sh, /*int ldim_S,*/double * Fh, /*int ldim_F,*/double * Gh/*, int ldim_G */, int p);


#endif // __TASK_TSMQR_H__
