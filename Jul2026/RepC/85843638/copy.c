#include <x86intrin.h>
#include "mnblas.h"

void mncblas_scopy(const int N, const float *X, const int incX, 
				 float *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  for (; ((i < N) && (j < N)) ; i = i + incX + 4, j = j + incY + 4)
	{
	  Y [j] = X [i] ;
	  Y [j+1] = X [i+1] ;
	  Y [j+2] = X [i+2] ;
	  Y [j+3] = X [i+3] ;
	}
  return ;
}

void mncblas_scopy_omp(const int N, const float *X, const int incX, 
				 float *Y, const int incY)
{
  register unsigned int j = 0 ;

  //#pragma omp for schedule(static)
  for (; (j < N) ; j = j + incY + 4)
	{
	  Y [j] = X [j] ;
	  Y [j+1] = X [j+1] ;
	  Y [j+2] = X [j+2] ;
	  Y [j+3] = X [j+3] ;
	}

  return ;
}

void mncblas_scopy_vec(const int N, const float *X, const int incX, 
				 float *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  __m128 v;

  for (; ((i < N) && (j < N)) ; i = i + incX + 4, j = j + incY + 4)
	{
	  v = _mm_load_ps(X+i);
	  _mm_store_ps (Y+i, v) ;
	}

  return ;
}

void mncblas_dcopy(const int N, const double *X, const int incX, 
				 double *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  for (; ((i < N) && (j < N)) ; i = i * incX + 2, j = j * incY + 2)
	{
	  Y [j] = X [i] ;
	  Y [j+1] = X [i+1] ;
	}
  return ;
}

void mncblas_dcopy_omp(const int N, const double *X, const int incX, 
				 double *Y, const int incY)
{

  #pragma omp for schedule(static)
  for (register unsigned int j = 0; j < N ; j = j + incY + 2)
	{
	  Y [j] = X [j] ;
	  Y [j+1] = X [j+1] ;
	}
  return ;
}

void mncblas_dcopy_vec(const int N, const double *X, const int incX, 
				 double *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  __m128d v;

  for (; ((i < N) && (j < N)) ; i = i + incX + 2, j = j + incY + 2)
	{
	  v = _mm_load_pd(X+i);
	  _mm_store_pd(Y+i, v) ;
	}

  return ;
}

void mncblas_ccopy_vec(const int N, const void *X, const int incX, 
							void *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  float *XP = (float *) X;
  float *YP = (float *) Y;

  for (; ((i < N*2) && (j < N*2)) ; i = i + incX + 4, j = j + incY + 4)
	{
	  _mm_store_ps(YP+j, _mm_load_ps(XP+i)) ;
	}

  return ;

}

void mncblas_ccopy(const int N, const void *X, const int incX, 
							void *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  float *XP = (float *) X;
  float *YP = (float *) Y;

  for (; ((i < N*2) && (j < N*2)) ; i += incX + 4, j += incY + 4){
		YP[j] = XP[i] ;
		YP[j+1] = XP[i+1] ;
		YP[j+2] = XP[i+2];
		YP[j+3] = XP[i+3];
	}

  return ;

}

void mncblas_ccopy_omp(const int N, const void *X, const int incX, 
							void *Y, const int incY)
{
  float *XP = (float *) X;
  float *YP = (float *) Y;

  #pragma omp for schedule(static)
  for (register unsigned int j = 0;j < N*2; j += incY + 4){
		YP[j] = XP[j] ;
		YP[j+1] = XP[j+1] ;
		YP[j+2] = XP[j+2];
		YP[j+3] = XP[j+3];
	}

  return ;

}

void mncblas_zcopy(const int N, const void *X, const int incX, 
							void *Y, const int incY)
{
	register unsigned int i = 0 ;
	register unsigned int j = 0 ;
	double *XP = (double *) X;
	double *YP = (double *) Y;

	for (; ((i < N*2) && (j < N*2)) ; i += incX + 2, j += incY + 2){
		YP[j] = XP[i] ;
		YP[j+1] = XP[i+1] ;
		YP[j+2] = XP[i+2];
		YP[j+3] = XP[i+3];
	}
}

void mncblas_zcopy_omp(const int N, const void *X, const int incX, 
							void *Y, const int incY)
{
	register unsigned int j = 0 ;
	double *XP = (double *) X;
	double *YP = (double *) Y;

	for (register unsigned int j = 0; j < N*2; j += incY + 2){
		YP[j] = XP[j] ;
		YP[j+1] = XP[j+1] ;
		YP[j+2] = XP[j+2];
		YP[j+3] = XP[j+3];
	}
}

void mncblas_zcopy_vec(const int N, const void *X, const int incX, 
							void *Y, const int incY)
{
	register unsigned int i = 0 ;
	register unsigned int j = 0 ;
	double *XP = (double *) X;
	double *YP = (double *) Y;

	for (; ((i < N*2) && (j < N*2)) ; i = i + incX + 2, j = j + incY + 2)
	{
	  _mm_store_pd(YP+i, _mm_load_pd(XP+i)) ;
	}

	return ;

}

