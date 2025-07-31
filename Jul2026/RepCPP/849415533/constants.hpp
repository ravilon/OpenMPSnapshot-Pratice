#pragma once

#include <string>

namespace Constants {
    // Physical constants and measurements in SI units

    constexpr double second = 1.0;
    constexpr double minute = 60.0;
    constexpr double hour = 3600.0;
    constexpr double day = 86400.0;
    constexpr double year = 3.15576E+7;
    constexpr double massSun = 1.98847E+30;
    constexpr double massMoon = 7.342E+22;
    constexpr double massEarth = 5.9722E+24;
    constexpr double meter = 1.0;
    constexpr double au = 1.495978707E+11; // astronomical unit
    constexpr double radiusEarth = 6371000.0;
    constexpr double distanceMoon = 3.844E+8;
    constexpr double parsec = 3.0856775814913673E+16;
    constexpr double gravityConstant = 6.6743E-11; // gravitational constant
    constexpr double lightSpeed = 2.99792458E+8;
    constexpr double pi = 3.14159265358979323846264338;
    constexpr double planckConstant = 6.62607015E-34;
    constexpr double kilogram = 1.0;
    constexpr double kilometer = 1000.0;

    double getConstantByName(const std::string& name);
};
