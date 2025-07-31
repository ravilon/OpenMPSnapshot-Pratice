#pragma once

#include "particle.hpp"
#include "unit_system.hpp"

#include <vector>
#include <string>
#include <filesystem>

class ParticleSystem {
    public:
        ParticleSystem(const std::filesystem::path& inputFilePath,
                       const std::shared_ptr<const UnitSystem> simulationUnitSystem);

        void simulate(const double fixedTimeStep,
                      const std::filesystem::path& outputDirPath,
                      const bool enableAdaptiveTimeStep, const double maxVelocityStep,
                      const double maxTime, const unsigned long maxIterations,
                      const double writeStatePeriod,
                      const std::string& integrationMethod);

    private:
        std::vector<Particle> particles;
        const std::string inputFileStem;
};
