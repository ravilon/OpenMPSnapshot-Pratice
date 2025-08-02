#include "qrca_kernels.h"
#include "qrca_utils.h"
#include <stdlib.h>


#define min( a, b ) ( (a) < (b) ? (a) : (b) )


/* Apply several block Householder transformations defined by (Ui,Si)
   to matrix C from left side:
     for i = 1, n_U/t
       C := ( I - Ui * Si * Ui' ) * C.
   Arguments:
     t                   Block size to use in the factorization.
     U  in      m_U x n_U     Matrix with the Householder vectors.
                              Defined by ( buff_U, ldim_U ).
     S  in      t x n_U  Matrix with triangular factors S.
                              Defined by ( buff_S, ldim_S ).
     C  in/out  m_U x n_C     Matrix to be applied the reflectors.
                              Defined by ( buff_C, ldim_C ).
*/

void dlarfb_task_hier( int t, int m_U, int n_U, int n_C, int skip, 
        double * buff_U, 
        double * buff_S, 
        double * buff_C) 
{
	n_C=n_U>>1;
	dlarfb_task(t, m_U, n_U, n_C, skip, buff_U, buff_S, buff_C);


	buff_C+=skip;
	buff_U+=skip;
	int m=m_U-skip;

  	double *buff_W = ( double * ) malloc( n_U * n_U * sizeof( double ) );
	

	buff_C=&buff_C[n_C*m_U];
	n_C=n_U-n_C;
	int mn_U = min( m_U, n_U );

  	int k;
	for( k = 0; k < mn_U; k += t ) {
		int b = min( t, mn_U - k );
		int m_U21 = m - k; 

    		double *buff_U11 = &( buff_U[ m_U * k + k] );          // U( k, k )
		double *buff_S1  = &( buff_S[ t * k + 0 ] );          // S( 0, k )
		double *buff_C1  = &( buff_C[ m_U * 0 + k ] );          // C( k,   0 )

		//printf("m_U21 %i n_U %i b%i nc %i\n",m_U21,n_U,b,n_C);

		dlarfb_("Left","Transpose","Forward","Columnwise",\
			&m_U21, &n_C, &b,\
			buff_U11, &m_U,\
			buff_S1, &t,\
			buff_C1, &m_U,\
			buff_W, &n_U);

  }

  free( buff_W );

#pragma omp taskwait
}

