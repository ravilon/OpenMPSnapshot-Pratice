//
// Simulação de propagação de vírus.
//
// Uso: omp_virus <tamanho-da-populacao> <nro. experimentos> <probab. maxima>
// <nro. threads>
#include <iostream>
#include "Random.h"
#include "Population.h"
#include <omp.h>

#include <cstdlib>
#include <string>

bool checkCommandLine(int argc, char **argv, int &size, int &trials, int &probs,
                      int &threads) {
    bool ambiente = true;
    if (argc > 1) {
        size = atoi(argv[1]);
    }
    if (argc > 2) {
        trials = atoi(argv[2]);
    }
    if (argc > 3) {
        probs = atoi(argv[3]);
    }
    if (argc > 4) {
        threads = atoi(argv[4]);
        ambiente = false;
    }
    return ambiente;
}

int main(int argc, char *argv[]) {
    // parâmetros dos experimentos
    int population_size = 30;
    int n_trials = 5000;
    int n_probs = 101;
    int n_threads = 1;

    double *percent_infected; // percentuais de infectados (saída)
    double *prob_spread;      // probabilidades a serem testadas (entrada)
    double prob_min = 0.0;
    double prob_max = 1.0;
    double prob_step;
    int base_seed = 100;

    bool env = checkCommandLine(argc, argv, population_size, n_trials, n_probs,
                                n_threads);
    try {
        // CÓDIGO COMPARTILHADO
        Random rand;
        prob_spread = new double[n_probs];
        percent_infected = new double[n_probs];
        prob_step = (prob_max - prob_min) / (double)(n_probs - 1);
        std::cout << "Probabilidade, Percentual Infectado" << std::endl;

        if (!env) { // EXECUÇÃO SETANDO THREADS POR PARAMETRO
            omp_set_num_threads(n_threads);
            Population *population = new Population(population_size);
            for (int ip = 0; ip < n_probs; ip++) {      
                prob_spread[ip] = prob_min + (double)ip * prob_step;
                percent_infected[ip] = 0.0;
                rand.setSeed(base_seed + ip);
                for (int it = 0; it < n_trials; it++) {
                    population->propagateUntilOut(population->centralPerson(),
                                                  prob_spread[ip], rand);
                    percent_infected[ip] += population->getPercentInfected();
                }
                percent_infected[ip] /= n_trials;
                std::cout << std::fixed << prob_spread[ip] << ", "
                          << percent_infected[ip] << std::endl;
            }
        } else { // EXECUÇÃO USANDO THREADS DO ENV
            const char *env_th = std::getenv("OMP_NUM_THREADS");
            n_threads = std::stoi(env_th); // Pega as Threads do primeiro nível
            Population **population = new Population *[n_threads];
            for (int i = 0; i < n_threads; i++)
                population[i] = new Population(population_size);

            omp_set_max_active_levels(2);
            omp_set_nested(true);
            for (int ip = 0; ip < n_probs; ip++) {
                prob_spread[ip] = prob_min + (double)ip * prob_step;
                percent_infected[ip] = 0.0;
                rand.setSeed(base_seed + ip);
#pragma omp parallel for schedule(dynamic) reduction(+ : percent_infected[ip])
                for (int it = 0; it < n_trials; it++) {
                    int id = omp_get_thread_num();
                    population[id]->propagateUntilOut(
                        population[id]->centralPerson(), prob_spread[ip], rand);
                    percent_infected[ip] +=
                        population[id]->getPercentInfected();
                }
                percent_infected[ip] /= n_trials;
                std::cout << std::fixed << prob_spread[ip] << ", "
                          << percent_infected[ip] << std::endl;
            }
        }

        delete[] prob_spread;
        delete[] percent_infected;
    } catch (std::bad_alloc) {
        std::cerr << "Erro: alocacao de memoria" << std::endl;
        return 1;
    }

    return 0;
}
