#pragma once

#include <random>
#include <string>
#include <vector>

#include "ga/individual.h"
#include "ga/logger.h"

#define DEFAULT_POPULATION 100
#define DEFAULT_STEP 5
#define SURVIVAL_RATE 0.5
#define DEFAULT_TIME_INTERVAL 10

#ifndef __cpp_lib_hardware_interference_size
namespace std {
constexpr std::size_t hardware_destructive_interference_size = 64;
}
#endif

/*
Base class for all genetic algorithm implementations.
This class serves as the foundation for implementing various genetic algorithms.
It provides common methods such as ...
*/
class BaseGA {
   public:
    BaseGA(std::string exp_name, int population_size = DEFAULT_POPULATION,
           int num_steps = DEFAULT_STEP, int thread_num = 1);
    ~BaseGA() = default;

    // executes the algorithm.
    void execute(int print_interval = DEFAULT_TIME_INTERVAL);

   protected:
    // cache aligned to avoid false-sharing
    struct alignas(std::hardware_destructive_interference_size) Padded19937 {
        std::mt19937 gen;
        Padded19937(unsigned int seed) : gen(seed) {}
    };

    std::string exp_name;
    int population_size;
    int current_population;
    int num_survivor;
    int num_steps;

    std::vector<Individual> population;

    int thread_num;
    std::vector<Padded19937> generators;

    Logger logger;

    // select 2 parents from suriviors using Roulette Wheel approach
    std::vector<int> select_parents(double sum, std::mt19937& generator);

   private:
    // Evaluate fitness score for all current population.
    virtual void evaluate_fitness() = 0;

    // Retain only the top n% of the population, various implementation depending on the GA model.
    virtual void selection_step() = 0;

    // Perform crossover and mutation to fill the population gap result from selection.
    virtual void update_population();
};