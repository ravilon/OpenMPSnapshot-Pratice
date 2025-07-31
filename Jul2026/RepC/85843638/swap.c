#include "mnblas.h"
#include <x86intrin.h>

void mncblas_sswap(const int N, float *X, const int incX, 
                 float *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  register float save ;
  
  for (; ((i < N) && (j < N)) ; i += incX, j+=incY)
    {
      save = Y [j] ;
      Y [j] = X [i] ;
      X [i] = save ;
    }

  return ;
}

void mncblas_sswap_omp(const int N, float *X, const int incX, 
                 float *Y, const int incY)
{
  register float save ;

  #pragma omp for schedule(static) private (save)
  for (register unsigned int j = 0; j<N ; j+=incY)
    {
      save = Y [j] ;
      Y [j] = X [j] ;
      X [j] = save ;
    }

  return ;
}

void mncblas_sswap_vec(const int N, float *X, const int incX, 
                 float *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  __m128 save, x, y;

  for (; ((i < N) && (j < N)) ; i += incX * 4, j+=incY * 4)
    {
      save = _mm_load_ps(Y+i) ;
      _mm_store_ps(Y+i, _mm_load_ps(X+i)) ;
      _mm_store_ps(X+i, save) ;
    }

  return ;
}

void mncblas_dswap(const int N, double *X, const int incX, 
                 double *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  register double save ;
  
  for (; ((i < N) && (j < N)) ; i += incX, j+=incY)
    {
      save = Y [j] ;
      Y [j] = X [i] ;
      X [i] = save ;
    }

  return ;
}

void mncblas_dswap_vec(const int N, double *X, const int incX, 
                 double *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  __m128d save, x, y;

  for (; ((i < N) && (j < N)) ; i += incX * 2, j+=incY * 2)
    {
      save = _mm_load_pd(Y+j) ;
      _mm_store_pd(Y+j, _mm_load_pd(X+i)) ;
      _mm_store_pd(X+i, save) ;
    }

  return ;
}

void mncblas_dswap_omp(const int N, double *X, const int incX, 
                 double *Y, const int incY)
{
  register double save ;

  #pragma omp for schedule(static) private(save)
  for (register unsigned int j = 0;j<N;j+=incY)
    {
      save = Y [j] ;
      Y [j] = X [j] ;
      X [j] = save ;
    }

  return ;
}

void mncblas_cswap(const int N, void *X, const int incX, 
		                    void *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  register vcomplexe save;
  float *XP = (float *) X;
  float *YP = (float *) Y;

  for (; ((i < N*2) && (j < N*2)) ; i += incX + 2, j+=incY + 2)
    {
      save.REEL = YP [j];
      save.IMAG = YP[j+1] ;
      YP[j] = XP[j];
      YP[j+1] = XP[j];
      XP [i] = save.REEL;
      XP[i+1] = save.IMAG;
    }

  return ;
}

void mncblas_cswap_omp(const int N, void *X, const int incX, 
                        void *Y, const int incY)
{
  register vcomplexe save;
  float *XP = (float *) X;
  float *YP = (float *) Y;

  #pragma omp for schedule(static) private (save)
  for (register unsigned int j = 0; j<N*2; j+=incY + 2)
    {
      save.REEL = YP [j];
      save.IMAG = YP[j+1] ;
      YP[j] = XP[j];
      YP[j+1] = XP[j];
      XP [j] = save.REEL;
      XP[j+1] = save.IMAG;
    }

  return ;
}

void mncblas_cswap_vec(const int N, void *X, const int incX, 
                        void *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  float *XP = (float *) X;
  float *YP = (float *) Y;

    __m128 saveX;

  for (; ((i < N*2) && (j < N*2)) ; i += incX + 4, j+=incY + 4)
    {
      saveX = _mm_load_ps(XP+i);
      _mm_store_ps(XP+i, _mm_load_ps(YP+i)) ;
      _mm_store_ps(YP+i, saveX) ;

    }

  return ;
}

void mncblas_zswap(const int N, void *X, const int incX, 
		                    void *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  register dcomplexe save;
  double *XP = (double *) X;
  double *YP = (double *) Y;

  for (; ((i < N*2) && (j < N*2)) ; i += incX + 2, j+=incY + 2)
    {
      save.REEL = YP [j];
      save.IMAG = YP[j+1] ;
      YP[j] = XP[j];
      YP[j+1] = XP[j];
      XP [j] = save.REEL;
      XP[j+1] = save.IMAG;
    }

  return ;
}

void mncblas_zswap_omp(const int N, void *X, const int incX, 
                        void *Y, const int incY)
{
  register dcomplexe save;
  double *XP = (double *) X;
  double *YP = (double *) Y;

  #pragma omp for schedule(static) private (save)
  for (register unsigned int j = 0; j < N*2;j+=incY + 2)
    {
      save.REEL = YP [j];
      save.IMAG = YP[j+1] ;
      YP[j] = XP[j];
      YP[j+1] = XP[j];
      XP [j] = save.REEL;
      XP[j+1] = save.IMAG;
    }

  return ;
}

void mncblas_zswap_vec(const int N, void *X, const int incX, 
                        void *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  double *XP = (double *) X;
  double *YP = (double *) Y;

    __m128d saveX;

  for (; ((i < N*2) && (j < N*2)) ; i += incX + 2, j+=incY + 2)
    {
      saveX = _mm_load_pd(XP+i);
      _mm_store_pd(XP+i, _mm_load_pd(YP+i)) ;
      _mm_store_pd(YP+i, saveX) ;

    }

  return ;
}

