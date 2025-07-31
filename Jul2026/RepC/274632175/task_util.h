#ifndef __TASK_UTIL_H__ 
#define __TASK_UTIL_H__ 


//#if BUILDING_LIBFOO && HAVE_VISIBILITY
#define LIBBBLAS_EXPORT __attribute__((__visibility__("default")))
//#else
//#define LIBFOO_DLL_EXPORTED
//#endif


#pragma omp task out([n]a)
extern LIBBBLAS_EXPORT void task_sclear(int n, float *a);

#pragma omp task out([n]a)
extern LIBBBLAS_EXPORT void task_dclear(int n, double *a);


#pragma omp task out([bm]A) label(sset) no_copy_deps
extern LIBBBLAS_EXPORT void task_sset(int bm, int bn, int m, int n, float v, float *A);

#pragma omp task out([bm]A) label(dset) no_copy_deps
extern LIBBBLAS_EXPORT void task_dset(int bm, int bn, int m, int n, double v, double *A);


#endif // __TASK_UTIL_H__ 
