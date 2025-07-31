#include <iostream>
#include <vector>
#include <algorithm>
#include <stack>
#include <climits>
#include <chrono>
#include <omp.h>

#include "../../tests/readfiles.h"

// `N` is the total number of total nodes on the graph or cities on the map
#ifdef SIZE
#undef N
#define N SIZE
#else
#define N 10
#endif

using namespace std;

struct tour_t;

int n;
int costMatrix[N][N];
tour_t *best_tour;

// tour data structure that stores
// - array storing the cities visited
// - the number of cities
// - the total distance of the tour
struct tour_t
{
    vector<int> cities;
    int cost;
};

/* Función que realiza la creación de un nuevo tour */
tour_t *newTour(vector<int> &cities, int cost)
{
    tour_t *node = new tour_t;
    node->cities = cities;
    node->cost = cost;
    return node;
}

/* Funcion para añadir una cuidad al tour y actualizar su costo */
void add_city(tour_t *&tour, int city)
{
    // Add city to the tour
    tour->cost += costMatrix[tour->cities.back()][city];
    tour->cities.push_back(city);
}

/* Funcion para eliminar una cuidad al tour y actualizar su costo */
void remove_city(tour_t *&tour, int city)
{
    // Remove last city from tour
    tour->cities.pop_back();
    tour->cost -= costMatrix[tour->cities.back()][city];
}

/* Función que realiza una copia del tour actual y lo añade a la pila, 
 * esto es realizado ya que al final del pseudocódigo la variable curr\_tour
 * es eliminada */
void push_copy(stack<tour_t *> &stack, tour_t *&tour)
{
    tour_t *tmp = newTour(tour->cities, tour->cost);
    stack.push(tmp);
}

/* Función que verifica si el costo actual del tour es menor al mejor costo */
bool is_best_tour(tour_t *&tour)
{
    // Add cost from last node to root node
    int last_mile_cost;
    last_mile_cost = tour->cost + costMatrix[tour->cities.back()][tour->cities.front()];

    if (last_mile_cost < best_tour->cost)
    {
        return true;
    }
    else
        return false;
}

/* Función que se encarga de actualizar la variable global best_tour y calcular se costo final */
void update_best_tour(tour_t *&tour)
{
    // Sum the cost back to the root node
    best_tour->cities = tour->cities;
    best_tour->cost = tour->cost + costMatrix[tour->cities.back()][tour->cities.front()];
}

/* determina si la ciudad ya ha sido visitado o no, 
 * en caso que si retorna true y en caso que no,
 * se determinar si el costo de este camino es menor 
 * que el mejor costo hasta el momento. */
bool feasible(tour_t *&tour, int nbr)
{
    bool lower_cost;
    int current_tour_cost;

    // Feasible if city has already been visited
    if (find(tour->cities.begin(), tour->cities.end(), nbr) != tour->cities.end())
    {
        return false;
    }
    // or it can lead to a least-cost tour
    current_tour_cost = tour->cost + costMatrix[tour->cities.back()][nbr];
    // Check if the new tour is better than the best current one
    lower_cost = (current_tour_cost < best_tour->cost);

    return lower_cost;
}

int main(int argc, char *argv[])
{

    /* Read Files and Set variables */
    int result_to_compare[N + 1];
    int result_cost_to_compare;
    read_matrix_and_result_from_file(argv[1], &n, &result_cost_to_compare, costMatrix, result_to_compare);

    /* Create stack */
    stack<tour_t *> root_stack;
    stack<tour_t *> my_stack;

    int thread_count = strtol(argv[2], NULL, 10);
    // Initialize threads
    omp_set_num_threads(thread_count);

    // Initialize best tour
    vector<int> best_tour_root;
    best_tour = newTour(best_tour_root, INT_MAX);

    // Initialize first tour
    vector<int> root_cities;
    root_cities.push_back(0);
    tour_t *root_tour = newTour(root_cities, 0);
    //root_stack.push(root);

    double start = omp_get_wtime();

    // Initialize thread-local best tour
#pragma omp parallel private(my_stack)
    {
#pragma omp for
        for (int i = 1; i < N; i++)
        {
            tour_t *local_tour = newTour(root_cities, 0);
            add_city(local_tour, i);
            my_stack.push(local_tour);
        }
        // Main loop
        while (!my_stack.empty())
        {
            tour_t *curr_tour = my_stack.top();
            my_stack.pop();
            if (curr_tour->cities.size() == n)
            {
                if (is_best_tour(curr_tour))
                {
#pragma omp critical
                    {
                        if (is_best_tour(curr_tour))
                        {
                            update_best_tour(curr_tour);
                        }
                    }
                }
            }
            else
            {
                for (int nbr = n - 1; nbr >= 1; nbr--)
                {
                    if (feasible(curr_tour, nbr))
                    {
                        add_city(curr_tour, nbr);
                        push_copy(my_stack, curr_tour);
                        remove_city(curr_tour, nbr);
                    }
                }
            }
            delete curr_tour;
        }
    }

    double stop = omp_get_wtime();

    int result_tsp[N + 1]; // use to set the result vecto to array

    // Output best tour
    std::cout << "Best tour is: " << endl;
    for (size_t i = 0; i < best_tour->cities.size(); i++) {
        std::cout << best_tour->cities[i] << " ";
        result_tsp[i] = best_tour->cities[i];
    }
    std::cout << best_tour->cities[0] << " ";
    result_tsp[N] = best_tour->cities[0];
    std::cout << std::endl;
    std::cout << "Best tour cost is: " << best_tour->cost << std::endl;

    // Print time taken
    double time = stop - start;
    print_time(time, true);

    // Test if the result is correct
    test(result_tsp, result_to_compare, best_tour->cost, result_cost_to_compare);

    return 0;
}