#include "Population.h"

Population::Population(int size) {
    this->size = size;
    pop = new Person *[size];
    #pragma omp parallel for shared(pop)
    for (int i = 0; i < size; i++)
        pop[i] = new Person[size];
    reset();
}

Population::~Population() {
    #pragma omp parallel for shared(pop)
    for (int i = 0; i < size; i++)
        delete[] pop[i];
    delete[] pop;
}

void Population::reset() {
#pragma omp parallel for shared(pop) collapse(2)
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            pop[i][j] = Uninfected;
}

PersonPosn Population::centralPerson() {
    PersonPosn p = {size / 2, size / 2};
    return p;
}

int Population::propagateUntilOut(PersonPosn sp, double prob_spread,
                                  Random &r) {
    int count;

    reset();
    pop[sp.i][sp.j] = Exposed;
    hasExposed = true;

    // Espalha enquanto tiverem pessoas expostas
    count = 0;
    while (hasExposed) {
        propagate(prob_spread, r);
        count++;
    }

    return count;
}

double Population::getPercentInfected() {
    int total = size * size - 1;
    int sum = 0;

// Calcula quantidade de pessoas infectadas
#pragma omp parallel for reduction(+ : sum) shared(pop) schedule(static)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (pop[i][j] == Infected) {
                sum++;
            }
        }
    }
    return ((double)(sum - 1) / (double)total);
}

// TODO: Paralelizar
void Population::propagate(double prob_spread, Random &r) {

// Pessoas expostas são infectadas pelo vírus
#pragma omp parallel for shared(pop) schedule(static)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (pop[i][j] == Exposed)
                pop[i][j] = Infected;
        }
    }
    hasExposed = false;
// Pessoas não infectadas são expostas ao vírus quando se aproximam de uma
// infectada
#pragma omp parallel for shared(pop, hasExposed) schedule(guided)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (pop[i][j] == Infected) {
                if (i != 0) { // pessoa ao norte
                    if (pop[i - 1][j] == Uninfected &&
                        virusSpreads(prob_spread, r)) {
#pragma omp critical
                        pop[i - 1][j] = Exposed;
                        hasExposed = true;
                    }
                }
                if (i != size - 1) { // pessoa ao sul
                    if (pop[i + 1][j] == Uninfected &&
                        virusSpreads(prob_spread, r)) {
#pragma omp critical
                        pop[i + 1][j] = Exposed;
                        hasExposed = true;
                    }
                }
                if (j != 0) { // pessoa a oeste
                    if (pop[i][j - 1] == Uninfected &&
                        virusSpreads(prob_spread, r)) {
#pragma omp critical
                        pop[i][j - 1] = Exposed;
                        hasExposed = true;
                    }
                }
                if (j != size - 1) { // pessoa a leste
                    if (pop[i][j + 1] == Uninfected &&
                        virusSpreads(prob_spread, r)) {
#pragma omp critical
                        pop[i][j + 1] = Exposed;
                        hasExposed = true;
                    }
                }
            }
        }
    }
}

bool Population::virusSpreads(double prob_spread, Random &r) {
    if (r.nextDouble() < prob_spread)
        return true;
    else
        return false;
}