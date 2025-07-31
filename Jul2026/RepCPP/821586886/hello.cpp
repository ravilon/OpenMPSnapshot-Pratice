#include <omp.h> //define funciton prototypes runtime library
#include <iostream>

int main(){
    #pragma omp parallel // Pragma - modifies compiler behaviour tells to use openmp to execute below block in parallel. 
    {
        int ID= omp_get_thread_num();
        std::cout<<"Hello"<<ID<std::endl<;
        std::cout<<"World"<<ID<<std::endl;
    } 
    return 0;
}