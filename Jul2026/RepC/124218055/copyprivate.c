#include <stdio.h>
#include <omp.h>
int main()
{
int rang;
float a;
#pragma omp parallel private(a,rang)
{
a = 92290.;
#pragma omp single copyprivate(a)
{
a = -92290.;
}
rang=omp_get_thread_num();
printf("Rang : %d ; A vaut : %f\n",rang,a);
}
return 0;
}