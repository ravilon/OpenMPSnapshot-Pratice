#include <stdio.h>
#include <omp.h>
int main()
{
int s, p;
s = 0, p = 1;
#pragma omp parallel
{
#pragma omp critical
{
s++;
p*=2;
}
}
printf("Somme et produit finaux : %d, %d\n",s,p);
return 0;
}