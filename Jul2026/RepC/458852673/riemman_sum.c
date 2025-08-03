#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double f(double x);
double p_riemman(double deltaX, int s, int n);
double l_riemman(double deltaX, int s, int n);

int main(int argc, char **argv){
if(!argv[1] || !argv[2] || !argv[3] || !argv[4] || !argv[5]){
printf("Enter the start, end, interval quantity, left/right, parallel? s/n when calling the command (e.g ./file.out a b n l/r s/n)\n");
exit(1);
}

int s = *argv[4] == 108 ? 0 : 1;

int a = atoi(argv[1]), b = atoi(argv[2]), n = atoi(argv[3]) * 1e8;

double psum = 0, rsum = 0;
double deltaX = (double)(b - a)/n;

int i = 0;

double start = omp_get_wtime();

if(*argv[5] == 110) rsum = l_riemman(deltaX, s, n);
else rsum = p_riemman(deltaX, s, n);

double end = omp_get_wtime();

printf("n: %d rsum: %f time: %f\n", n, rsum, end - start);

return 0;
}

double f(double x){
return pow(x, 2) + 2;
}

double p_riemman(double deltaX, int s, int n){
double rsum = 0, psum = 0;
int i = 0;

#pragma omp parallel shared(rsum, deltaX, s, n) firstprivate(i, psum)
{
#pragma omp for nowait
// #pragma omp for nowait schedule(dynamic, 160)
for(i = 0; i < n; i++) psum += f((i + s) * deltaX) * deltaX;

#pragma omp critical
rsum += psum;
}

return rsum;
}

double l_riemman(double deltaX, int s, int n){
double rsum = 0;

for(int i = 0; i < n; i++) rsum += f((i + s) * deltaX) * deltaX;

return rsum;
}