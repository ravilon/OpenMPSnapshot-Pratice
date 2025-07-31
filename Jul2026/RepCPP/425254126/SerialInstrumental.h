#pragma once

#include "algo/interfaces/Instrumental.h"

#include <utility>


class SerialInstrumental : public Instrumental {
protected:
    double h;
    vec x;
    vec A, C, B;

public:
    SerialInstrumental() :
            Instrumental(),
            h(1 / static_cast<double>(N)),
            x(std::move(this->getGridNodes())) {}

    explicit SerialInstrumental(size_t n) :
            SerialInstrumental(n, vec(n, 0), vec(n, 0), vec(n, 0)) {}

    SerialInstrumental(const vec& a, vec c, vec b) :
            SerialInstrumental(a.size() + 1, a, std::move(c), std::move(b)) {}

    SerialInstrumental(size_t n, vec a, vec c, vec b) :
            Instrumental(n),
            h(1 / static_cast<double>(N)),
            x(this->getGridNodes()),
            A(std::move(a)), C(std::move(c)), B(std::move(b)) {}

    // Preparing user data for serial computing
    void prepareData() override;

    // Checking for ...
    bool checkData() const override;

    // Getting a grid with nodes
    vec getGridNodes();

    // Getting protected fields
    std::tuple<double, vec, vec, vec, vec> getAllFields();

    /*
     * Creating a tridiagonal matrix with dimension @N x @N
     *
     * side lower diagonal = @a
     * side upper diagonal = @b
     * algo diagonal	   = @c
    */
    matr createMatr();
};