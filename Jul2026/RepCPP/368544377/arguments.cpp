#include "omp.h"

#include <iostream>
#include <string>

/**
  * @brief File demoing different ways to pass data to a parallel code block
  * There are different ways to pass data from a global scope (in relation to the scope the parallel block of code) to change these default settings - you should explicitly specify what option you want:
    * To use data declared in a global scope directly (ie shared variable): shared(<var_name>)
    * To create a local, uninitialised copy of var (ie just its datatype): private(<var_name>)
        * This value is destroyed at the end of the code block's scope
    * To create a local, initialised copy of var (ie just its datatype): firstprivate(<var_name>)
        * This value is destroyed at the end of the code block's scope
  * The way to pass data BACK from a code block, asides from shared access: lastprivate(<var_name>)
        * This value is NOT destroyed, rather it transfers the value of a private from the last iteration to the global variable
        * NOTE: this is limited purely to sections / loops
  * These specifiers are placed at the end of the `#pragma` statement of a given code block ( e.g. #pragma omp parallel private(var) )
  */

int main()
{
    int var = 13;

    #pragma omp parallel num_threads(1)
    {
        std::cout << "implicit shared (DEFAULT): chances are this number is 13: " << var << std::endl ; // reason: var is referencing the global variable
    }
    // OR
    #pragma omp parallel num_threads(1) shared(var)
    {
        std::cout << "explicit shared: chances are this number is 13: " << var << std::endl ; // reason: var is referencing the global variable
    }

    #pragma omp parallel private(var) num_threads(1)
    {
        std::cout << "private: chances are this number is not 13: " << var << std::endl ; // reason: var is NOT initialised, its datatype is just passed etc.
    }

    #pragma omp parallel firstprivate(var) num_threads(1)
    {
        std::cout << "firstprivate: chances are this number is 13: " << var << std::endl ; // reason: var is initialised to the same value and type of the variable passed in
    }

    #pragma omp parallel for lastprivate(var) num_threads(1)
    for(int i = 0 ; i < 1 ; ++i)
    {
        std::cout << "lastprivate: chances are this number is 13: " << var << std::endl ; // reason: var is initialised to the same value and type of the variable passed in
    }
    std::cout << "GLOBAL CONTEXT: Chances are lastprivate change the value from 13: " << var << std::endl ;

    //
	return 0 ;
}
