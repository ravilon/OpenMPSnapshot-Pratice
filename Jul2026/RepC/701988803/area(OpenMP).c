#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

long millisecond()
{
   struct timeval tv;
   gettimeofday(&tv, NULL);
   return(1000*tv.tv_sec + tv.tv_usec/1000) ;
}

double f(double x)
{
   double temp;
   temp = x*x+2;
   return x/(temp*temp*temp);
}

main(int argc, char *argv[])
{
   long N, i;
   double a, b, area, local_area, dx, x;
   long start, elapsed;
   int nt, tid;

   
   if (argc != 3) {
      printf("argument error\n");
      exit(1);
   }

   N = atol(argv[1]);

   // sequential calculation of area
   start = millisecond();
   a = 0.0;
   b = 2.0; 
   dx = (b-a)/(double)N;

   area = 0.0;

   x = a;
   for (i=0; i<N; i++) {
      area += 0.5*(f(x)+f(x+dx))*dx;
      x += dx;
   }

   elapsed = millisecond() - start;

   printf("%5.10lf\n", area);
   printf("elapsed time: %ld milliseconds\n", elapsed);
   printf("GFLOPS: %10.2f\n", (N*16.0/(elapsed/1000.0))/1000000000.0);

   nt = atoi(argv[2]);

   // parallel(openMP) calculation of area
   /* FILL IN THIS BLANK */
   omp_set_num_threads(nt);
   #pragma omp parallel private(i, x, tid, local_area) shared(dx, nt, area)
   {
      tid = omp_get_thread_num();
      x = ((b-a)/nt)*tid;

      #pragma omp for
      for(i = 0; i < N; i++) {
         local_area += 0.5*(f(x)+f(x+dx))*dx;
         x += dx;
      }
      #pragma omp atomic
      area += local_area;
   }


   printf("area(openMP): %5.10f\n", area);
   printf("elapsed time(openMP): %ld milliseconds\n", elapsed);
   printf("GFLOPS(openMP): %5.2f\n", (N*12.0/(elapsed/1000.0))/1000000000.0);

   exit(0);
}
