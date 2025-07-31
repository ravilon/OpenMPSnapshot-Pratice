#pragma once
#include <vector>

namespace numath {
    namespace systemsOfEquations {

        /**
         * Solves a system of N equations by using Doolittle's LU Factoring
         * @param A Matrix of coefficients
         * @param b Vector of independent terms
         */
        std::vector<double> doolittleMethod(std::vector<std::vector<double>> A, std::vector<double> b);

        /**
         * Fills the matrices L and U with their initial values
         * @param L Uninitialized L matrix
         * @param U Uninitialized U matrix
         */
        void __initializeMatrixDM(std::vector<std::vector<double>> &L,std::vector<std::vector<double>> &U);

        /**
         * Method that factors matrix A into two triangular matrices L and U
         * @param A Matrix of coefficients
         * @param L Initialized L matrix
         * @param U Initialized U matrix
         * @param N Equation system size
         */
        void __LUFactoringDM(std::vector<std::vector<double>> &A, std::vector<std::vector<double>> &L, std::vector<std::vector<double>> &U,  int N);

        /**
         * Finds the value of every unknown of the system.
         * 
         * @param L Factored L matrix
         * @param b Vector of independent terms
         */
        std::vector<double> __forwardSubstitutionDM(std::vector<std::vector<double>> &L, std::vector<double> &b);

        /**
         * Finds the value of every unknown of the system.
         * 
         * @param U Factored U matrix
         * @param z Transition vector
         */
        std::vector<double> __backwardSubstitutionDM(std::vector<std::vector<double>> &U, std::vector<double> &z);


        /**
         * Prints any needed matrix
         * @param matrix Matrix to be printed
         */
        void toStringMatrixD(std::vector<std::vector<double>> &matrix);

        /**
         * Printss an uncomplete matrix with unknown values
         * @param matrix Matrix to be printed
         * @param name Matrix name to show unknown items
         */
        void toStringIncMatrixD(std::vector<std::vector<double>> &matrix, char name);

    }
}