#include <stdio.h>
#include <omp.h>

int main()
{
     int x=10;
     #pragma omp parallel firstprivate(x) num_threads(1)
     {
          x+=5;
          printf("Value of x inside parallel region =%d\n",x);
     }
     printf("Value of x outside parallel region =%d\n",x);

     int y=10;
     /*lastprivate is not defined for omp parallel*/
     /*The meaning of lastprivate, is to assign "the sequentially last iteration of the associated loops,
      or the lexically last section construct [...] to the original list item."
      Hence, there it no meaning for a pure parallel construct. It would not be a good idea to use a meaning like 
     "the last thread to exit the parallel construct" - that would be a race condition.*/
     #pragma omp lastprivate(y) num_threads(1)
     {
          for(int j=5;j>=0;j--)
          {
               y-=1;
          }
          printf("Value of y inside parallel region =%d\n",y);
     }
     printf("Value of y outside parallel region =%d\n",y);
}
