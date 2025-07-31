#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]){

FILE* f = fopen("scale.txt", "a");

long long int Ntests = atoll(argv[1]);

double tic = omp_get_wtime();

long long int Ninside = 0;
long long n;
double estpi = 0;

struct drand48_data randBuffer;  

#pragma omp parallel
srand48_r(omp_get_num_threads(),&randBuffer);

//Tricky pragma statement
#pragma omp parallel for private(n, randBuffer) shared(Ntests) reduction(+:Ninside)
for(n=0;n<Ntests;++n){
double x;
double y;
drand48_r(&randBuffer, &x);
drand48_r(&randBuffer, &y);

//Same expression as serial here
if(sqrt(pow(x,2) + pow(y,2)) < 1){
Ninside++;
}
}

//Same expression as serial code here
estpi = 4*(double)Ninside / (double)Ntests;

double toc = omp_get_wtime();

double elapsedTime = toc - tic;

fprintf(f, "estPi = %lf\n", estpi);
fprintf(f, "dt = %f\n", elapsedTime);
fclose(f);
}
