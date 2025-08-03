# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <omp.h>
# include <getopt.h>

int main ( int argc, char *argv[] );
void r8_mxm ( int l, int m, int n );
double r8_uniform_01 ( int *seed );

int dimension_size = 1024; //Default: smallest size for test eval
int num_threads = 1; //Default: serial execution

/* Please modify for GPU Experiments */
/* @@@ Shahadat Hossain (SH) March 12, 2018 */
/******************************************************************************/


int main ( int argc, char **argv )
/******************************************************************************/
/*
Purpose:

<<< SH:  Skeletal c code for performing dense matrix times matrix. >>>
<<<      a = b*c where a, b, c are matrices of size n X n          >>>


Licensing:

This code is distributed under the GNU LGPL license. 

Modified:

@@@ Shahadat Hossain (SH) Nov 08, 2019 

*/
{
char opt;

while((opt = getopt(argc, argv, "n:t:")) != -1)
{
switch (opt)
{
case 'n':
dimension_size = atoi(optarg);
break;
case 't':
num_threads = atoi(optarg);
break;
default:
printf("Nothing was passed.");
}
}

if (num_threads <= 0)
{
num_threads = 1; //default to 1
}
if (dimension_size%2 !=0)
{
dimension_size = 1024; //default to smallest eval sz
}


int id;
int l;
int m;
int n;

printf ( "\n" );
printf ( "Dense MXM\n" );
printf ( "  C/OpenMP version.\n" );
printf ( "\n" );
printf ( "  Matrix multiplication tests.\n" );

/*  @@@ SH Note 1a:

You must read in the dimension of the matrix and the number of threads
from the command line.
*/
printf ( "\n" );
printf ( "  Number of processors available = %d\n", omp_get_num_procs ( ) );
printf ( "  Number of threads =              %d\n", omp_get_max_threads ( ) );
printf ( "  Matrix dimension size =          %dx%d\n", dimension_size, dimension_size );
printf ( "  Chosen number of threads =       %d\n", num_threads );

/*  
@@@ SH Note 1b: 

These values need to be read in from command line. Assume that 
l = m = n. 
*/
l = dimension_size;
m = dimension_size;
n = dimension_size;

r8_mxm( l, m, n ); // call the matrix multiplication routine

/*
Terminate.
*/
printf ( "\n" );
printf ( "Dense MXM:\n" );
printf ( "  Normal end of execution.\n" );

return 0;
}
/******************************************************************************/

void r8_mxm ( int l, int m, int n )

/******************************************************************************/
/*
Purpose:

R8_MXM carries out a matrix-matrix multiplication in double precision real  arithmetic.

Discussion:

a(lxn) = b(lxm) * c(mxn).

Licensing:

This code is distributed under the GNU LGPL license. 

Modified:

Shahadat Hossain (SH) Nov 15, 2014

Parameters:

Input: int l, m, n, the dimensions that specify the sizes of the
a, b, and c matrices.
*/
{


double **a;
double **b;
double **c;
int i;
int j;
int k;
int K_;
int I_;
int J_;
int temp;
int ops;
double rate;
int seed;
double time_begin;
double time_elapsed;
double time_stop;

int s = 18;
/*
Allocate the storage for matrices.
*/
a = ( double ** ) calloc ( l, sizeof ( double ) );
b = ( double ** ) calloc ( l , sizeof ( double ) );
c = ( double ** ) calloc ( m , sizeof ( double ) );
for ( i = 0; i < l ; i++)
a[i] = (double *) calloc (n, sizeof (double));
for ( i = 0; i < l ; i++)
b[i] = (double *) calloc (m, sizeof (double));
for ( i = 0; i < m ; i++)
c[i] = (double *) calloc (n, sizeof (double));
/*
Assign randomly generated values to the input matrices B and C.
*/
seed = 123456789;

for ( k = 0; k < l ; k++ )
for ( j = 0; j < m; j++)
{
b[k][j] = r8_uniform_01 ( &seed );
}

for ( k = 0; k < m ; k++ )
for (j = 0; j < n; j++)
{
c[k][j] = r8_uniform_01 ( &seed );
}
/*
Compute a = b * c.
*/
/* 
@@@ SH Note 2a:
— The timer function omp_get_wtime() is used to record wallclock time.

— The parallel directive given in the code below is for information only. 
Your job is to try and use different directives as well as loop rearrangement 
and other code optimization that you have learnt in the course to obtain  
maximum sequential and parallel performance. 
*/ 

time_begin = omp_get_wtime ( );

omp_set_num_threads(num_threads);
# pragma omp parallel \
shared ( a, b, c, l, m, n ) \
private ( i, j, k, K_, I_, J_, temp)

# pragma omp for
for(I_=0;I_<n;I_+= s)
{
for(J_=0;J_<n;J_+= s)
{
for(i=0;i<n;i++)
{
for(j = I_; j<((I_+s)>n?n:(I_+s)); j++)
{
temp = 0;
for(k = J_; k<((J_+s)>n?n:(J_+s)); k++){
temp += b[i][k] * c[k][j];
}
a[i][j] += temp;
}
}
}
}

time_stop = omp_get_wtime ( );
/*
Generate Report.

@@@ SH Notes 3b :
In the reporting part, you should also compute and report parallel efficiency.
*/
ops = l * n * ( 2 * m );
time_elapsed = time_stop - time_begin;
rate = ( double ) ( ops ) / time_elapsed / 1000000.0;

printf ( "\n" );
printf ( "R8_MXM matrix multiplication timing.\n" );
printf ( "  A(LxN) = B(LxM) * C(MxN).\n" );
printf ( "  L = %d\n", l );
printf ( "  M = %d\n", m );
printf ( "  N = %d\n", n );
printf ( "  Floating point OPS roughly %d\n", ops );
printf ( "  Elapsed time dT = %f\n", time_elapsed );
printf ( "  Rate = MegaOPS/dT = %f\n", rate );

free ( a );
free ( b );
free ( c );

return;
}
/******************************************************************************/

double r8_uniform_01 ( int *seed )

/******************************************************************************/
/*
Purpose:

R8_UNIFORM_01 is a unit pseudorandom double precision real number R8.

Discussion:

This routine implements the recursion

seed = 16807 * seed mod ( 2**31 - 1 )
unif = seed / ( 2**31 - 1 )

The integer arithmetic never requires more than 32 bits,
including a sign bit.

Licensing:

This code is distributed under the GNU LGPL license. 

Modified:

11 August 2004

Author:

John Burkardt

Reference:

Paul Bratley, Bennett Fox, Linus Schrage,
A Guide to Simulation,
Springer Verlag, pages 201-202, 1983.

Bennett Fox,
Algorithm 647:
Implementation and Relative Efficiency of Quasirandom
Sequence Generators,
ACM Transactions on Mathematical Software,
Volume 12, Number 4, pages 362-376, 1986.

Parameters:

Input/output, int *SEED, a seed for the random number generator.

Output, double R8_UNIFORM_01, a new pseudorandom variate, strictly between
0 and 1.
*/
{
int k;
double r;

k = *seed / 127773;

*seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

if ( *seed < 0 )
{
*seed = *seed + 2147483647;
}

r = ( double ) ( *seed ) * 4.656612875E-10;

return r;
}
