///
/// \file src/test/fista.cpp
/// \brief Implementation of the FISTA class test suite.
/// \author Philippe Ganz <philippe.ganz@gmail.com> 2017-2019
/// \version 1.0.1
/// \date August 2019
/// \copyright GPL-3.0
///

#include "test/fista.hpp"

namespace alias
{
namespace test
{
namespace fista
{

bool SmallExample()
{
    std::cout << "FISTA test with small data : " << std::endl << std::endl;

    double A_data[12] = {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0};
    const MatMult<double> A(Matrix<double>(A_data, 12, 3, 4), 3, 4);
    double u_data[3] = {3.0,2.0,1.0};
    const Matrix<double> u(u_data, 3, 3, 1);
    double b_data[3] = {1.0,1.0,2.0};
    const alias::Matrix<double> b(b_data, 3, 3, 1);
    alias::fista::poisson::Parameters<double> options;
    options.log_period = 20;
    double tols[3] = {1e-4, 1e-8, 1e-12};

    double expected_data1[4] = {0.939214191176937, 0.0, 0.0, -0.650697826008454};
    double expected_data2[4] = {0.973320232506235, 0.0, 0.0, -0.674615621567228};
    double expected_data3[4] = {0.973633428618360, 0.0, 0.0, -0.674833246032450};

    const Matrix<double> expected_result[3] = {{expected_data1, 4, 4, 1},{expected_data2, 4, 4, 1},{expected_data3, 4, 4, 1}};

    bool fista_test = true;

    for( int i = 0; i < 3; ++i )
    {
        std::cout << "Running with A = " << A.Data();
        std::cout << ", b = " << b;
        std::cout << "and u = " << u;

        options.tol = tols[i];
        Matrix<double> actual_result = alias::fista::poisson::Solve(A, u, b, 1.0, options);

        std::cout << "Result from MATLAB computation" << expected_result[i];
        std::cout << "Result from this computation" << actual_result;

        double relative_error = std::abs((actual_result - expected_result[i]).Norm(two)) / std::abs(expected_result[i].Norm(two));

        bool local_result = (relative_error < 100*options.tol);
        fista_test = fista_test && local_result;

        std::cout << std::endl << (local_result ? "Success" : "Failure") << ", achieved ";
        std::cout << relative_error << " relative norm error with a tol of " << tols[i] << "." << std::endl;
    }

    return fista_test;
}

void Time(size_t length)
{
    std::cout << "FISTA test with big data : " << std::endl << std::endl;

    std::default_random_engine generator;
    generator.seed(123456789);
    std::normal_distribution<double> distribution(100.0,10.0);
    size_t test_height = length*length;
    size_t test_width = length;

    double* A_data = new double[test_height*test_width]; // destroyed when A is destroyed
    #pragma omp parallel for simd
    for( size_t i = 0; i < test_height*test_width; ++i )
    {
        A_data[i] = distribution(generator);
    }
    MatMult<double> A(Matrix<double>(A_data, test_height, test_width), test_height, test_width);

    double* u_data = new double[test_height]; // destroyed when u is destroyed
    #pragma omp parallel for simd
    for( size_t i = 0; i < test_height; ++i )
    {
        u_data[i] = distribution(generator);
    }
    Matrix<double> u(u_data, test_height, 1);

    double* b_data = new double[test_height]; // destroyed when b is destroyed
    #pragma omp parallel for simd
    for( size_t i = 0; i < test_height; ++i )
    {
        b_data[i] = distribution(generator);
    }
    Matrix<double> b(b_data, test_height, 1);

    alias::fista::poisson::Parameters<double> options;
    options.tol = 1e-8;
    options.log_period = 1;

    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    start = std::chrono::high_resolution_clock::now();

    alias::fista::poisson::Solve(A, u, b, 1.0, options);

    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end-start;
    std::cout << std::endl << std::endl << "Time for FISTA solver with " << test_height;
    std::cout << "x" << test_width << " double matrix : " << std::defaultfloat << elapsed_time.count();
    std::cout << " seconds" << std::endl << std::endl;
}

} // namespace matrix
} // namespace test
} // namespace alias
