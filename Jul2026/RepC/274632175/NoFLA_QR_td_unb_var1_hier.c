#include <stdlib.h>
#include "NoFLA_QR_td_unb_var1.h"
#include "qrca_kernels.h"


#define min( a, b ) ( (a) < (b) ? (a) : (b) )


/* Compute QR factorization of [ U; D ], where U is upper triangular, and
   returns factors tau into vector t, and triangular factors into matrix S.
   Arguments:
     U  in/out  m_U x n_U     Top part of matrix to be factorized.
                              Upper triangular. Defined by ( buff_U, ldim_U ).
     D  in/out  m_D x n_U     Bottom part of matrix to be factorized.
                              Dense. Defined by ( buff_D, ldim_D ).
     t  out     n_U           Vector with tau scalars.
                              Defined by ( buff_t ).
     S  out     mn_U x mn_U   Matrix with triangular factors S.
                              Defined by ( buff_S, ldim_S ). 
                              mn_U = min( m_U, n_U ).
*/


int NoFLA_QR_td_unb_var1_hier( int m_U, int n_U, int m_D, 
                          double * buff_U, int ldim_U,
                          double * buff_D, int ldim_D,
                          double * buff_t,
                          double * buff_S, int ldim_S ) {
	int     mn_U, m_D_plus_1, j, n_wt2, i_one = 1;
	double  * buff_w, d_one = 1.0, d_zero = 0.0, d_minus_one = -1.0;

	mn_U       = min( m_U, n_U );
	m_D_plus_1 = m_D + 1;

	buff_w = ( double * ) malloc( n_U * sizeof( double ) );


	for( j = 0; j < mn_U; j++ ) {
		n_wt2 = n_U - j - 1;

		dlarfg_( & m_D_plus_1, & buff_U[ j * ldim_U + j ], 
     			& buff_D[ j * ldim_D + 0 ], & i_one, & buff_t[ j ] );


		/* Update the rest of U and D by applying the Householder transform:
 			wt2  := u12t + d1' * D2;
 			u12t := u12t - tau1      * wt2;
 			D2   := D2   - tau1 * d1 * wt2; */
		dcopy_( & n_wt2, & buff_U[ (j+1) * ldim_U + j ], & ldim_U,
                     & buff_w[ 0 ], & i_one );

		dgemv_( "Transpose", & m_D, & n_wt2, 
    			& d_one, & buff_D[ (j+1) * ldim_D + 0 ], & ldim_D,  
	     		& buff_D[ j * ldim_D + 0 ], & i_one,
    			& d_one, & buff_w[ 0 ], & i_one );

		buff_t[ j ] = - buff_t[ j ];

		daxpy_( & n_wt2, & buff_t[ j ],
    			& buff_w[ 0 ], & i_one,
    			& buff_U[ (j+1) * ldim_U + j ], & ldim_U );

		dger_( & m_D, & n_wt2, & buff_t[ j ],
   			& buff_D[ j * ldim_D + 0 ], & i_one,
   			& buff_w[ 0 ], & i_one,
   			& buff_D[ (j+1) * ldim_D + 0 ], & ldim_D );
 
		buff_t[ j ] = - buff_t[ j ];
	}


	free( buff_w );


	// Update the columns of S
	NoFLA_QR_td_unb_var1_updateS(mn_U, m_D, buff_D, ldim_D, buff_t, buff_S, ldim_S); 

#pragma omp taskwait
    
	return 0;
}
