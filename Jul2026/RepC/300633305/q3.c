#include "omp.h"
#include<stdio.h>
void​ main()
{
/* Parallel region begins here */
double​ time;
int​ i;
double​ x[​ 100​ ],y[​ 100​ ];
for​ (i=​ 0 ​ ;i<​ 900​ ;i++)
{
x[i] = i/​ 1000​ ;
y[i] = ​ 2 ​ *i/​ 10000​ ;
}
double​ a = ​ 7.52423​ ;
double​ start;
double​ ​ end​ ;
start = omp_get_wtime();
#pragma omp for
for​ (i = ​ 0 ​ ; i < ​ 900​ ; i++)y[i] = a*x[i] + y[i];
end​ = omp_get_wtime();
printf(​ "Work took %f seconds\n"​ , ​ end​ - start);
}
