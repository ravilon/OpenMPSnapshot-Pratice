#include "ga/openmp_ga.h"

#include <omp.h>

#include <algorithm>
#include <iostream>

OpenmpGA::OpenmpGA(std::string exp_name, int population_size, int num_steps, int thread_num)
    : BaseGA(exp_name, population_size, num_steps, thread_num) {
    omp_set_num_threads(thread_num);
#ifdef DEBUG
    std::cout << "initialize new OpenmpGA with " << thread_num << " threads" << std::endl;
#endif
}

void OpenmpGA::evaluate_fitness() {
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
#pragma omp for schedule(dynamic)
        for (int i = 0; i < population_size; i++) {
            population[i].evaluate(generators[thread_id].gen);
        }
    }
}

void OpenmpGA::selection_step() {
    std::sort(population.begin(), population.end(),
              [](Individual &a, Individual &b) -> bool { return a.fitness > b.fitness; });
    current_population = num_survivor;
}

void OpenmpGA::update_population() {
    double sum = 0;

#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < num_survivor; i++) {
        sum += population[i].fitness;
    }
    std::shuffle(population.begin(), population.begin() + num_survivor, generators[0].gen);

    int target_num = population_size - current_population;
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int thread_workload = target_num / thread_num;
        const int start_pos = current_population + thread_id * thread_workload;
        if (thread_id == thread_num - 1) thread_workload += target_num % thread_num;
        std::vector<Individual> local_offsprings;
        while (local_offsprings.size() < thread_workload) {
            std::vector<int> par = select_parents(sum, generators[thread_id].gen);
            Individual &father = population[par[0]];
            Individual &mother = population[par[1]];

            for (auto &child : father.crossover(generators[thread_id].gen, mother)) {
                child.mutate(generators[thread_id].gen);
                local_offsprings.emplace_back(std::move(child));
                if (local_offsprings.size() == thread_workload) break;
            }
        }

#pragma omp critical
        for (int i = 0; i < thread_workload; i++) {
            population[start_pos + i] = std::move(local_offsprings[i]);
        }
    }
}
