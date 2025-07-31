#pragma once

#include <iostream>
#include <vector>


class AbstractSweepMethod {
public:
    // Calling sweep method
    virtual std::vector<double> run() = 0;
};
