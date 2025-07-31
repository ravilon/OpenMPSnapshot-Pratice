#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "consts.h"
#include "pngwriter.h"

unsigned long get_time()
{
       struct timeval tp;
       gettimeofday(&tp, NULL);
       return tp.tv_sec * 1000000 + tp.tv_usec;
}

int main(int argc, char **argv)
{
       png_data *pPng = png_create(IMAGE_WIDTH, IMAGE_HEIGHT);

       double x, y, x2, y2, cx, cy;
       cy = MIN_Y;

       double fDeltaX = (MAX_X - MIN_X) / (double)IMAGE_WIDTH;
       double fDeltaY = (MAX_Y - MIN_Y) / (double)IMAGE_HEIGHT;

       long nTotalIterationsCount = 0;
       unsigned long nTimeStart = get_time();

       long i, j, n;

       #pragma omp parallel for default(none) shared(fDeltaX, fDeltaY, pPng) collapse(2) reduction(+ : nTotalIterationsCount)
       for (int j = 0; j < IMAGE_HEIGHT; j++)
       {
              for (int i = 0; i < IMAGE_WIDTH; i++)
              {
                     double cx = MIN_X + i * fDeltaX;
                     double cy = MIN_Y + j * fDeltaY;
                     double x = cx;
                     double y = cy;
                     double x2 = x * x;
                     double y2 = y * y;
                     int n = 0;

                     // Innermost loop remains serial (due to collapse(2))
                     for (n = 1; (x2 + y2) < 4 && n < MAX_ITERS; n++)
                     {
                            y = 2 * x * y + cy;
                            x = x2 - y2 + cx;
                            x2 = x * x;
                            y2 = y * y;
                     }

                     nTotalIterationsCount += n;

                     int c = ((long)n * 255) / MAX_ITERS;
                     png_plot(pPng, i, j, c, c, c);
              }
       }
       unsigned long nTimeEnd = get_time();

       // print benchmark data
       printf("Total time:                 %g millisconds\n",
              (nTimeEnd - nTimeStart) / 1000.0);
       printf("Image size:                 %ld x %ld = %ld Pixels\n",
              (long)IMAGE_WIDTH, (long)IMAGE_HEIGHT,
              (long)(IMAGE_WIDTH * IMAGE_HEIGHT));
       printf("Total number of iterations: %ld\n", nTotalIterationsCount);
       printf("Avg. time per pixel:        %g microseconds\n",
              (nTimeEnd - nTimeStart) / (double)(IMAGE_WIDTH * IMAGE_HEIGHT));
       printf("Avg. time per iteration:    %g microseconds\n",
              (nTimeEnd - nTimeStart) / (double)nTotalIterationsCount);
       printf("Iterations/second:          %g\n",
              nTotalIterationsCount / (double)(nTimeEnd - nTimeStart) * 1e6);
       // assume there are 8 floating point operations per iteration
       printf("MFlop/s:                    %g\n",
              nTotalIterationsCount * 8.0 / (double)(nTimeEnd - nTimeStart));
       char fileName[] = "mandel.png";
       png_write(pPng, fileName);
       return 0;
}
