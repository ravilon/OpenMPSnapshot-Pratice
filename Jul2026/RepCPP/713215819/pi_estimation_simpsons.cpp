#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <random>
void parallel_pi(int num_steps)
{
    double step;
    int i;
    double x, pi, sum = 0.0;
    step = 1.0 / (double)num_steps;
    double start = omp_get_wtime();
    #pragma omp parallel for private(x) 
    for (i = 0; i < num_steps; i++) {
        x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    pi = step * sum;
    double end = omp_get_wtime();
    std::cout << "pi = " << pi << std::endl;
    std::cout << "time = " << end - start << std::endl;
}
void parallel_pi_atomic(int num_steps)
{
    double step;
    int i;
    double x, pi, sum = 0.0;
    step = 1.0 / (double)num_steps;
    double start = omp_get_wtime();
    #pragma omp parallel for private(x) 
    for (i = 0; i < num_steps; i++) {
        x = (i + 0.5) * step;
        #pragma omp atomic
        sum += 4.0 / (1.0 + x * x);
    }
    pi = step * sum;
    double end = omp_get_wtime();
    std::cout << "pi = " << pi << std::endl;
    std::cout << "time = " << end - start << std::endl;
}

void parallel_pi_critical(int num_steps)
{

    double step;
    int i;
    double x, pi, sum = 0.0;
    step = 1.0 / (double)num_steps;
    double start = omp_get_wtime();
    #pragma omp parallel for private(x) 
    for (i = 0; i < num_steps; i++) {
        x = (i + 0.5) * step;
        #pragma omp critical
        sum += 4.0 / (1.0 + x * x);
    }
    pi = step * sum;
    double end = omp_get_wtime();
    std::cout << "pi = " << pi << std::endl;
    std::cout << "time = " << end - start << std::endl;
}
void parallel_pi_reduction(int num_steps)
{
    double step;
    int i;
    double x, pi, sum = 0.0;
    step = 1.0 / (double)num_steps;
    double start = omp_get_wtime();
    #pragma omp parallel for private(x) reduction(+:sum)
    for (i = 0; i < num_steps; i++) {
        x = (i + 0.5) * step;
        //#pragma omp critical
        sum += 4.0 / (1.0 + x * x);
    }
    pi = step * sum;
    double end = omp_get_wtime();
    std::cout << "pi = " << pi << std::endl;
    std::cout << "time = " << end - start << std::endl;
}

void serial_pi(int num_steps)
{
    double step;
    int i;
    double x, pi, sum = 0.0;
    step = 1.0 / (double)num_steps;
    double start = omp_get_wtime();
    for (i = 0; i < num_steps; i++) {
        x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    pi = step * sum;
    double end = omp_get_wtime();
    std::cout << "pi = " << pi << std::endl;
    std::cout << "time = " << end - start << std::endl;
}



int main() {
    int num_steps = 100;
    std::cout<<"Dhivyesh R K"<<std::endl;
    std::cout<<"2021BCS0084"<<std::endl;
    std::cout<<"Parallel pi no clauses"<<std::endl;
    parallel_pi(num_steps);
    std::cout<<"Parallel pi atomic"<<std::endl;
    parallel_pi_atomic(num_steps);
    std::cout<<"Parallel pi critical"<<std::endl;
    parallel_pi_critical(num_steps);
    std::cout<<"Parallel pi reduction"<<std::endl;
    parallel_pi_reduction(num_steps);
    std::cout<<"Serial pi"<<std::endl;
    serial_pi(num_steps);

    std::cout<< "-------------------------------------"<<std::endl;
    num_steps = 500;
    std::cout<<"Dhivyesh RK"<<std::endl;
    std::cout<<"2021BCS0084"<<std::endl;
    std::cout<<"Parallel pi no clauses"<<std::endl;
    parallel_pi(num_steps);
    std::cout<<"Parallel pi atomic"<<std::endl;
    parallel_pi_atomic(num_steps);
    std::cout<<"Parallel pi critical"<<std::endl;
    parallel_pi_critical(num_steps);
    std::cout<<"Parallel pi reduction"<<std::endl;
    parallel_pi_reduction(num_steps);
    std::cout<<"Serial pi"<<std::endl;
    serial_pi(num_steps);
    std::cout<< "-------------------------------------"<<std::endl;
    num_steps = 1000;
    std::cout<<"Dhivyesh  RK"<<std::endl;
    std::cout<<"2021BCS0084"<<std::endl;
    std::cout<<"Parallel pi no clauses"<<std::endl;
    parallel_pi(num_steps);
    std::cout<<"Parallel pi atomic"<<std::endl;
    parallel_pi_atomic(num_steps);
    std::cout<<"Parallel pi critical"<<std::endl;
    parallel_pi_critical(num_steps);
    std::cout<<"Parallel pi reduction"<<std::endl;
    parallel_pi_reduction(num_steps);
    std::cout<<"Serial pi"<<std::endl;
    serial_pi(num_steps);
    std::cout<< "-------------------------------------"<<std::endl;
    num_steps = 100000000;
    std::cout<<"Dhivyesh R K"<<std::endl;
    std::cout<<"2021BCS0084"<<std::endl;
    std::cout<<"Parallel pi no clauses"<<std::endl;
    parallel_pi(num_steps);
    std::cout<<"Parallel pi atomic"<<std::endl;
    parallel_pi_atomic(num_steps);
    std::cout<<"Parallel pi critical"<<std::endl;
    parallel_pi_critical(num_steps);
    std::cout<<"Parallel pi reduction"<<std::endl;
    parallel_pi_reduction(num_steps);
    std::cout<<"Serial pi"<<std::endl;
    serial_pi(num_steps);
    
    
    return 0;
}
