#pragma once

#include "algo/interfaces/Instrumental.h"


class ParallelInstrumental : public Instrumental {
private:
    // is the @num prime
    static bool isPrime(int num);

    // Finding all the divisors of @num
    static vec findDivisors(int num);

protected:
    size_t threadNum, blockSize, interSize;

public:
    ParallelInstrumental() : ParallelInstrumental(5, 0, -1, -1) {}

    ParallelInstrumental(size_t n, size_t tN) : Instrumental(n) {
        this->prepareData(n, tN);
    }

    ParallelInstrumental(size_t n, size_t threadNum, size_t blockSize, size_t interSize) : Instrumental(n),
              threadNum(threadNum), blockSize(blockSize), interSize(interSize) {
        this->setParallelOptions();
    }

    void setParallelOptions() const;

    // Preparing user data for parallel computing
    void prepareData(size_t n, size_t threadNum);

    void prepareData() override;

    // Checking for multiplicity of @N and @THREAD_NUM
    bool checkData() const override;

    /*
     * Creating a tridiagonal matrix with dimension @N x @N
     *
     * side lower diagonal = 1
     * side upper diagonal = 2
     * algo diagonal	   = 3
    */
    matr createThirdDiagMatrI();

    // Creating a matrix with random values from 0 to 100 with dimension @N x @N
    matr createThirdDiagMatrRand();

    // Creating a vector from 0 to @N with dimension @N
    vec createVecN();

    // Creating a vector with random values from 0 to 100 with dimension @N
    vec createVecRand();

    matr createNewMatr(vec a_, vec c_, vec b_, pairs kappa_, pairs gamma_);

    vec createNewVec(vec phi_, pairs mu_);
};