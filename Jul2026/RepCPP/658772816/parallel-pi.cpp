#include <omp.h>
#include <stdio.h>
#include <random>

int main(){
    unsigned int num_threads, points_per_thread, npoints = 10000000;
    unsigned int sum;
    double rand_no_x, rand_no_y;
    unsigned int seed = 42;
    
    #pragma omp parallel private(rand_no_x, rand_no_y, num_threads, points_per_thread) shared(npoints) reduction(+ : sum) num_threads(8)
    {
        num_threads = omp_get_num_threads();
        points_per_thread = npoints / num_threads;
        sum = 0;
        for(int i = 0; i < points_per_thread; i++){
            rand_no_x = (double)(rand_r(&seed)) / (double)RAND_MAX;
            rand_no_y = (double)(rand_r(&seed)) / (double)RAND_MAX;
            if (((rand_no_x) * (rand_no_x) + (rand_no_y) * (rand_no_y)) < 1.0){
                sum += 1;
            }
        }
    }

    printf("Points sampled inside circle=%d | Total points sampled=%d.\nResulting pi = %f\n", sum, npoints, 4.0 * (float)sum/(float)npoints);
}