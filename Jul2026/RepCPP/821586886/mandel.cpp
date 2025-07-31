#include <omp.h>
#include <iostream>
#include <cmath>

#define NPOINTS 1000
#define MAXITER 1000


struct d_complex{
   double r;
   double i;
};

void testpoint(d_complex& z); 

struct d_complex c;
int numoutside = 0;

omp_lock_t numoutside_lock;

int main(){
    int i, j;
    double area, error, eps  = 1.0e-5;
    double start_time, run_time;

    int n_threads;

    std::cout<<"Enter Number Of Threads:";
    std::cin>>n_threads;

    omp_set_num_threads(n_threads);
    //   Loop over grid of points in the complex plane which contains the Mandelbrot set,
    //   testing each point to see whether it is inside or outside the set.
    omp_init_lock(&numoutside_lock);

    start_time = omp_get_wtime();
    #pragma omp parallel for collapse(2) default(none) shared(eps, numoutside) private(c, i, j)
    {
        for (i=0; i<NPOINTS; i++){
            for (j=0; j<NPOINTS; j++){
                c.r = -2.0+2.5*(double)(i)/(double)(NPOINTS)+eps;
                c.i = 1.125*(double)(j)/(double)(NPOINTS)+eps;
                testpoint(c);
            }
        }
    }
    run_time = omp_get_wtime() - start_time;
    omp_destroy_lock(&numoutside_lock);


    // Calculate area of set and error estimate and output the results
    area=2.0*2.5*1.125*(double)(NPOINTS*NPOINTS-numoutside)/(double)(NPOINTS*NPOINTS);
    error=area/(double)NPOINTS;

    std::cout<<"Area of Mandlebrot set = "<<area<<" +/- "<<error<<" in "<<run_time<<" seconds"<<std::endl;
    std::cout<<"Correct answer should be around 1.510659"<<std::endl;

    return 0;
}

void testpoint(d_complex& c){
    // Does the iteration z=z*z+c, until |z| > 2 when point is known to be outside set
    // If loop count reaches MAXITER, point is considered to be inside the set

    int iter;
    double temp;
    d_complex z = {0.0, 0.0};

    for (iter=0; iter<MAXITER; iter++){
        temp = (z.r*z.r)-(z.i*z.i)+c.r;
        z.i = z.r*z.i*2+c.i;
        z.r = temp;

        if ((z.r*z.r+z.i*z.i)>4.0){
            omp_set_lock(&numoutside_lock);
            //#pragma omp atomic
            numoutside++;
            omp_unset_lock(&numoutside_lock);
            break;
        }
    }
}