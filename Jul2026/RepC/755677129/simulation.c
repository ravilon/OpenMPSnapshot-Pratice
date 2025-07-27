#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

#include "constants.h"
#include "grid.h"

void simulation(char ***grid, int32_t N, int64_t *max_counts, int32_t *max_generations, int32_t num_generations)
{
    // Temporary grid to hold the next generation
    char ***next_grid = (char ***)malloc(N * sizeof(char **));
    if (next_grid == NULL)
    {
        fprintf(stderr, "Failed to allocate memory for next_grid\n");
        exit(1);
    }

    for (int32_t x = 0; x < N; x++)
    {
        next_grid[x] = (char **)malloc(N * sizeof(char *));
        if (next_grid[x] == NULL)
        {
            fprintf(stderr, "Failed to allocate memory for next_grid\n");
            exit(1);
        }
        next_grid[x][0] = (char *)calloc(N * N, sizeof(char));
        if (next_grid[x][0] == NULL)
        {
            fprintf(stderr, "Failed to allocate memory for next_grid\n");
            exit(1);
        }
        for (int32_t y = 1; y < N; y++)
            next_grid[x][y] = next_grid[x][0] + y * N;
    }

    // Print for debugging - Initial grid (generation 0)
    // printf("Generation 0    ------------------------------\n");
    // print_grid(grid, N);

    int64_t initial_species_counts[N_SPECIES + 1] = {0};

    #pragma omp parallel for shared(grid, N) reduction(+ : initial_species_counts[ : N_SPECIES + 1])
    for (int32_t x = 0; x < N; x++)
    {
        for (int32_t y = 0; y < N; y++)
        {
            for (int32_t z = 0; z < N; z++)
            {
                int16_t species = grid[x][y][z];
                if (species > 0)
                    initial_species_counts[species]++;
            }
        }
    }

    // Update the maximum counts and generations
    for (int16_t s = 1; s <= N_SPECIES; s++)
    {
        max_counts[s] = initial_species_counts[s];
        max_generations[s] = 0;
    }

    // Perform simulation for the specified number of generations
    for (int32_t gen = 1; gen <= num_generations; gen++)
    {
        int64_t species_counts[N_SPECIES + 1] = {0};

        // Iterate over each cell in the grid
        #pragma omp parallel for shared(grid, N) reduction(+ : species_counts[ : N_SPECIES + 1])
        for (int32_t x = 0; x < N; x++)
        {
            for (int32_t y = 0; y < N; y++)
            {
                for (int32_t z = 0; z < N; z++)
                {
                    // Count the number of live neighbors for the current cell
                    int16_t alive_neighbors = 0;
                    // Determine the majority species of the neighbors
                    int16_t neighbor_species_counts[N_SPECIES + 1] = {0};

                    for (int8_t i = -1; i <= 1; i++)
                    {
                        int32_t nx = x + i;
                        nx = nx >= N ? 0 : nx < 0 ? N - 1
                                                  : nx;

                        for (int8_t j = -1; j <= 1; j++)
                        {
                            int32_t ny = y + j;
                            ny = ny >= N ? 0 : ny < 0 ? N - 1
                                                      : ny;

                            for (int8_t k = -1; k <= 1; k++)
                            {
                                if (i == 0 && j == 0 && k == 0)
                                    continue; // Skip the current cell

                                int32_t nz = z + k;
                                nz = nz >= N ? 0 : nz < 0 ? N - 1
                                                          : nz;

                                int16_t species = grid[nx][ny][nz];
                                if (species > 0)
                                {
                                    alive_neighbors++;
                                    neighbor_species_counts[species]++;
                                }
                            }
                        }
                    }

                    int16_t current_species = grid[x][y][z];

                    // Apply the rules of the Game of Life
                    if (current_species > 0)
                    {
                        if (alive_neighbors <= 4 || alive_neighbors > 13)
                            next_grid[x][y][z] = 0; // Cell dies
                        else
                        {
                            next_grid[x][y][z] = current_species; // Cell survives
                            species_counts[current_species]++;
                        }
                    }
                    else
                    {
                        if (alive_neighbors >= 7 && alive_neighbors <= 10)
                        {
                            // Determine the majority species
                            int16_t max_species = 0;
                            int16_t max_count = 0;
                            for (int16_t s = 1; s <= N_SPECIES; s++)
                            {
                                if (neighbor_species_counts[s] > max_count)
                                {
                                    max_species = s;
                                    max_count = neighbor_species_counts[s];
                                }
                            }
                            next_grid[x][y][z] = max_species; // Cell becomes alive with majority
                            species_counts[max_species]++;
                        }
                        else
                        {
                            next_grid[x][y][z] = 0; // Cell remains dead
                        }
                    }
                }
            }
        }

        // Swap the current grid with the next grid
        char ***temp = grid;
        grid = next_grid;
        next_grid = temp;

        // Update the maximum counts and generations
        for (int16_t s = 1; s <= N_SPECIES; s++)
        {
            if (species_counts[s] > max_counts[s])
            {
                max_counts[s] = species_counts[s];
                max_generations[s] = gen;
            }
        }

        // Print for debugging
        // printf("Generation %d    ------------------------------\n", gen);
        // print_grid(grid, N);
    }
}
