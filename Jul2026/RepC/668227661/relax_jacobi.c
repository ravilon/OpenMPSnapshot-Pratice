/*
 * relax_jacobi.c
 *
 * Jacobi Relaxation
 *
 */

#include "heat.h"

#ifdef NON_BLOCKING
double inner_relax_jacobi( algoparam_t *param, double **u1, double **utmp1,
		   unsigned sizex, unsigned sizey )
{
  int i, j;
  double *help,*u, *utmp,factor=0.5;

  utmp=*utmp1;
  u=*u1;
  double unew, diff, sum=0.0;

int istart = 2, iend = sizey - 2;
int jstart = 2, jend = sizex - 2;

  for( i=istart; i< iend; i++ ) {
  	int ii=i*sizex;
  	int iim1=(i-1)*sizex;
  	int iip1=(i+1)*sizex;

    for( j=jstart; j< jend; j++ ){
       utmp[ ii+ j] = 0.25 * (u[ ii+(j-1) ]+
        		            u[ ii+(j+1) ]+
        		            u[ iim1+j ]+
        		            u[ iip1+j ]);
		    diff = utmp[ii + j] - u[ii + j];
		    sum += diff * diff;
  
       }

  }
 

  return(sum);
}

double outer_relax_jacobi( algoparam_t *param, double **u1, double **utmp1,
         unsigned sizex, unsigned sizey )
{
  int i, j;
  double *help,*u, *utmp,factor=0.5;

  utmp=*utmp1;
  u=*u1;
  double unew, diff, sum=0.0;

  i = 1;
  int ii=i*sizex;
  int iim1=(i-1)*sizex;
  int iip1=(i+1)*sizex;
  //top row
  for( j=1; j< sizex -1; j++ )
	{
      utmp[ ii+ j] = 0.25 * (u[ ii+(j-1) ]+
                        u[ ii+(j+1) ]+
                        u[ iim1+j ]+
                        u[ iip1+j ]);
        diff = utmp[ii + j] - u[ii + j];
        sum += diff * diff;
  }

  //bottom row
  i = sizey - 2;
  ii=i*sizex;
  iim1=(i-1)*sizex;
  iip1=(i+1)*sizex;
  for( j=1; j< sizex -1; j++ )
	{
      utmp[ ii+ j] = 0.25 * (u[ ii+(j-1) ]+
                        u[ ii+(j+1) ]+
                        u[ iim1+j ]+
                        u[ iip1+j ]);
        diff = utmp[ii + j] - u[ii + j];
        sum += diff * diff;
  }

  //left col
  for( i=2; i< sizey -2; i++ )
	{
      j = 1;
      ii=i*sizex;
      iim1=(i-1)*sizex;
      iip1=(i+1)*sizex;
      utmp[ ii+ j] = 0.25 * (u[ ii+(j-1) ]+
                        u[ ii+(j+1) ]+
                        u[ iim1+j ]+
                        u[ iip1+j ]);
      diff = utmp[ii + j] - u[ii + j];
      sum += diff * diff;
  }

  //right col
  for( i=2; i< sizey -2; i++ )
	{
      j = sizex - 2;
      ii=i*sizex;
      iim1=(i-1)*sizex;
      iip1=(i+1)*sizex;
      utmp[ ii+ j] = 0.25 * (u[ ii+(j-1) ]+
                        u[ ii+(j+1) ]+
                        u[ iim1+j ]+
                        u[ iip1+j ]);
      diff = utmp[ii + j] - u[ii + j];
      sum += diff * diff;
  }
  
  *u1=utmp;
  *utmp1=u;
  
  return(sum);
}

#else
double relax_jacobi( double **u1, double **utmp1,
         unsigned sizex, unsigned sizey )
{
  int i, j;
  double *help,*u, *utmp,factor=0.5;

  utmp=*utmp1;
  u=*u1;
  double unew, diff, sum=0.0;

#pragma omp parallel for reduction(+:sum) private(diff) schedule(static) 
  for( i=1; i< sizey-1; i++ ) {
  	int ii=i*sizex;
  	int iim1=(i-1)*sizex;
  	int iip1=(i+1)*sizex;
#pragma ivdep
    for( j=1; j<sizex-1; j++ ){
       utmp[ ii+ j] = 0.25 * (u[ ii+(j-1) ]+
        		            u[ ii+(j+1) ]+
        		            u[ iim1+j ]+
        		            u[ iip1+j ]);
		    diff = utmp[ii + j] - u[ii + j];
		    sum += diff * diff;
       }
  }

  *u1=utmp;
  *utmp1=u;

  return(sum);
}
#endif

