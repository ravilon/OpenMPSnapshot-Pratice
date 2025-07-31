#pragma once

#include <omp.h>

#include <iostream>
#include <vector>
#include <tuple>
#include <random>
#include <functional>
#include <algorithm>
#include <numeric>

#define print() printf("---\n");
#define printd(a) printf("%s = %f\n", #a, a);
#define printi(a) printf("%s = %d\n", #a, a);

#define EPS 0.0001;

using matr = std::vector<std::vector<double>>;
using vec = std::vector<double>;
using pairs = std::pair<double, double>;
using str = std::string;


class Instrumental {
protected:
	size_t N, node;
	vec u, v;

public:
	Instrumental() : Instrumental(5) {}

	explicit Instrumental(size_t n) : N(n), node(n + 1), u(node), v(node) {}

	void setN(size_t n);

	void setUV(vec& u_, vec& v_);

	// Preparing user data for parallel computing
	virtual void prepareData();

	// Checking for multiplicity of @N and @THREAD_NUM
	virtual bool checkData() const;

	// Printing a vector @a with @name
	static void printVec(const vec& a, const str& name);

	// Printing a matrix @a with @name
	static void printMatr(const matr& a, const str& name);

	// Calculating the discrepancy
	double calcR(const vec& x, const vec& b) const;

	// Calculating of the error estimate of the scheme
	double calcZ() const;

	// Matrix-vector multiplication : @A x @b
	static vec calcMatrVecMult(const matr& A, const vec& b);

	// Getting protected fields
	std::tuple<size_t, size_t, vec, vec> getAllFields() const;

    // Compare of two doubles
    static bool compareDouble(double a, double b);

    // Comparison of two matrices
    static bool compareMatr(const matr& a, const matr& b);

    // Comparison of two vectors
    static bool compareVec(const vec& a, const vec& b);
};