/*
  The iterations of the k and j loops are collapsed into one loop with a larger iteration
  space, and that loop is then divided among the threads in the current team. 

  The variables in the collapse clause are implicitly private! So variabls k and j can be omitted from the private clause.

  The sequential execution of the iterations in the k and j loops determines the order of
the iterations in the collapsed iteration space. This implies that in the sequentially last
iteration of the collapsed iteration space, k will have the value 2 and j will have the
value 3. Since klast and jlast are lastprivate, their values are assigned by the
sequentially last iteration of the collapsed k and j loop. This example prints: 2 3.
*/

#include <stdio.h>
#include <omp.h>
void test()
{
 int j, k, jlast, klast;
 #pragma omp parallel
 {
 #pragma omp for collapse(2) lastprivate(jlast, klast)
 for (k=1; k<=2; k++)
 for (j=1; j<=3; j++)
 {
 jlast=j;
 klast=k;
 }
 #pragma omp single
 printf("%d %d\n", klast, jlast);
 }
}
int main()
{
test();
return 0;
}
