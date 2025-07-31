#include "config.hpp"
#include "util.hpp"
#include "particle_system.hpp"

#include <iostream>
#include <filesystem>
#include <memory>

#define CONFIG_PATH "../config.txt"

int main() {
    const Config config = Config::load(CONFIG_PATH);

    std::cout << "Simulations started at: " << getDateTimeString(false, 0) << std::endl;
    std::cout << "Output directory: " << config.outputDirPath << '\n' << std::endl;

    const auto inputFileEntries = getFileEntries(config.inputFilesDirPath);
    const size_t inputFileCount = inputFileEntries.size();
    const std::shared_ptr<const UnitSystem> sharedUnitSystem
        = std::make_shared<const UnitSystem>(config.unitSystem);
    size_t progress = 0;

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (const auto& fileEntry : inputFileEntries) {
        ParticleSystem particleSystem(fileEntry.path(), sharedUnitSystem);

        particleSystem.simulate(config.fixedTimeStep, config.outputDirPath,
                                config.enableAdaptiveTimeStep, config.maxVelocityStep,
                                config.maxTime, config.maxIterations,
                                config.writeStatePeriod, config.integrationMethod);

#ifdef _OPENMP
    #pragma omp critical
#endif
        {
            progress++;
            printProgress(progress, inputFileCount);
        }
    }

    return 0;
}
