#pragma once

#include "unit_system.hpp"

#include <filesystem>
#include <string>

class Config {
    public:
        const UnitSystem unitSystem;
        const std::filesystem::path outputDirPath;
        const std::filesystem::path inputFilesDirPath;
        const double fixedTimeStep;
        const double maxVelocityStep;
        const bool enableAdaptiveTimeStep;
        const double maxTime;
        const unsigned long maxIterations;
        const double writeStatePeriod;
        const std::string integrationMethod;

        static Config load(const std::filesystem::path& configPath);

    private:
        Config(const UnitSystem& unitSystem, const std::filesystem::path& outputDir,
               const std::filesystem::path& inputFilesDir, const double fixedTimeStep,
               const double maxVelocityStep, const bool enableAdaptiveTimeStep,
               const double maxTime, const unsigned long maxIterations,
               const double writeStatePeriod, const std::string& integrationMethod);
};
