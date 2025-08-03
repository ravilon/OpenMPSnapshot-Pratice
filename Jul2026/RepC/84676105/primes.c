/* primes.c
* --------
*
* Developed by George Z. Zachos
* Initial program developed by VVD.
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define UPTO 10000000


/* Global declarations and definitions */
long int count,      /* number of primes */
lastprime;  /* the last prime found */


void serial_primes(long int n) {
long int i, num, divisor, quotient, remainder;

if (n < 2) return;
count = 1;                         /* 2 is the first prime */
lastprime = 2;

for (i = 0; i < (n-1)/2; ++i) {    /* For every odd number */
num = 2*i + 3;

divisor = 1;
do
{
divisor += 2;                  /* Divide by the next odd */
quotient  = num / divisor;
remainder = num % divisor;
} while (remainder && divisor <= quotient);  /* Don't go past sqrt */

if (remainder || divisor == num) /* num is prime */
{
count++;
lastprime = num;
}
}
}


void openmp_primes(long int n) {
long int i, num, divisor, quotient, remainder;

if (n < 2) return;
count = 1;
lastprime = 2;


/* Scheduling policy controlled via OMP_SCHEDULE e.v. */
#pragma omp parallel for private(num,divisor,quotient,remainder)  reduction(max:lastprime) reduction(+:count)  schedule(runtime)
for (i = 0; i < (n-1)/2; ++i) {
num = 2*i + 3;

divisor = 1;
do
{
divisor += 2;
quotient  = num / divisor;
remainder = num % divisor;
} while (remainder && divisor <= quotient);

if (remainder || divisor == num)
{
count++;
lastprime = num;
}
}

}


int main(void)
{
double   start_time, elapsed_time;

//	printf("Serial and parallel prime number calculations:\n");
#if 0
/* Start timing */
start_time = omp_get_wtime();
serial_primes(UPTO);
/* End timing */
elapsed_time = omp_get_wtime() - start_time;
printf("[serial] count = %ld, last = %ld (time = %f)\n",
count, lastprime, elapsed_time);
#endif
/* Start timing */
start_time = omp_get_wtime();
openmp_primes(UPTO);
/* End timing */
elapsed_time = omp_get_wtime() - start_time;
printf("[openmp] count = %ld, last = %ld ( time = %f )\n",
count, lastprime, elapsed_time);

return EXIT_SUCCESS;
}
