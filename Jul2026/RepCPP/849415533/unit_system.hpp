#pragma once

#include <string>

class UnitSystem {
    public:
        const std::string id;
        const double unitLength, unitMass, unitTime, unitVelocity;
        const double gravityConstant; // gravitational constant in this unit system

        UnitSystem(const std::string& id);

        double convertLength(const double inputLength,
                             const UnitSystem& inputUnitSystem) const;
        double convertMass(const double inputMass,
                           const UnitSystem& inputUnitSystem) const;
        double convertTime(const double inputTime,
                           const UnitSystem& inputUnitSystem) const;
        double convertVelocity(const double inputVelocity,
                               const UnitSystem& inputUnitSystem) const;
};
