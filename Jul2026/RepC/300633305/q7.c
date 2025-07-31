#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include<omp.h>
#define SEED 35791246
void​ ​ main​ ()
{
int​ niter=​ 0 ​ ;
double​ x,y;
int​ i,count=​ 0 ​ ; ​ /* # of points in the 1st quadrant of unit
circle */
double​ z;
double​ pi;
rintf​ ( ​ "Enter the number of iterations used to estimate
p
pi: "​ );
scanf​ ( ​ "%d"​ ,&niter);
/* initialize random numbers */
srand(SEED);
count=​ 0 ​ ;
double​ start;
double​ end;
start = omp_get_wtime();
#pragma omp parallel private(i,x,y,z)
#pragma omp for schedule(static)
for​ ( i=​ 0 ​ ; i<niter; i++) {
x = (​ double​ )rand()/RAND_MAX;
y = (​ double​ )rand()/RAND_MAX;
z = x*x+y*y;
if​ (z<=​ 1 ​ ) count++;
}
end = omp_get_wtime();
printf​ ( ​ "Work took %f seconds\n"​ , end - start);
pi=(​ double​ )count/niter*​ 4 ​ ;
printf​ ( ​ "# of trials= %d , estimate of pi is %g
\n"​ ,niter,pi);
}
