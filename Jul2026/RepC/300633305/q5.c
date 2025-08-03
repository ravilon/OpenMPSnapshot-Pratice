#include "omp.h"
#include<stdio.h>
static​ ​ long​ num_steps = ​ 100000​ ;
double​ ​ step​ ;
void​ main ()
{
int​ i;
double​ x, pi, sum = ​ 0.0​ ,aux;
step​ = ​ 1.0​ /(​ double​ )num_steps;
double​ start;
double​ ​ end​ ;
start = omp_get_wtime();
#pragma omp parallel private(i,x,aux) shared(sum)
{
#pragma omp for schedule(static)
for​ (i=​ 0 ​ ; i<num_steps; i=i+​ 1 ​ ){
x=(i+​ 0.5​ )*​ step​ ;
aux=​ 4.0​ /(​ 1.0​ +x*x);
#pragma omp critical
sum = sum + aux;
}
}end​ = omp_get_wtime();
printf(​ "Work took %f seconds\n"​ , ​ end​ - start);
pi = ​ step​ * sum;
}
