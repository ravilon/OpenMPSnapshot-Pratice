#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

main(){

int n=9, i,a,b[n];

for(i=0;i<n;i++) b[i]=-1;

#pragma omp parallel
{
#pragma omp single
{
printf("Introduce valor de inicializacion a:");
scanf("%d",&a);
printf("Single ejecutada por el thread%d\n",omp_get_thread_num());
}
#pragma omp for
for(i=0;i<n;i++) b[i]=a;
}
printf("Después de la región parallel:\n");
#pragma omp single
{
for(i=0;i<n;i++){
printf("Single ejecutada por el thread%d\n",omp_get_thread_num());
printf("b[%d]=%d\t",i,b[i]);
}
}
printf("\n");
}