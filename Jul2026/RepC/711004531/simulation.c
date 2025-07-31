#include "include/simulation.h"

int calculate_next_position (int g, int x, int y, int p)
{
    return (g + x + y) % p;
}

void enumerate_positions (int *adj, int value) 
{
    // Enumerates starting from north the possible movement positions {0, 1, 2, 3} ==> {N, E, S, W}
    for (int i = 0; i < 4; i++) 
    {
        if (adj[i] == -1) 
        {
            adj[i] = value;
            break;
        }
    }
}

void empty_cells (int x, int y, int north, int south, int east, int west,
                  environment *config, int *adj, unsigned int *available, int type)
{
    // Checks if the adjacent cell is in bounds and its type to evaluate movement
    if (north >= 0 && north < config->r && config->spaces[north][y].type == type)
    {
        enumerate_positions(adj, 0);
        (*available)++;
    }
    if (east >= 0 && east < config->c && config->spaces[x][east].type == type)
    {
        enumerate_positions(adj, 1);
        (*available)++;
    }
    if (south >= 0 && south < config->r && config->spaces[south][y].type == type)
    {
        enumerate_positions(adj, 2);
        (*available)++;
    }
    if (west >= 0 && west < config->c && config->spaces[x][west].type == type)
    {
        enumerate_positions(adj, 3);
        (*available)++;
    }
}

void move_animal_to_adj (environment *config, cell **spaces, int x, int y,
                         int orientation, int type, bool vertical)
{
    // If the movement is in the vertical, performs operations only on the matrix x coordinate (row)
    if (vertical)
    {
        if (type == FOX)
        {
            // The stats of the current analyzed animal are transferred for the adjacent cell in the ecosystem
            spaces[orientation][y].type = config->spaces[x][y].type;
            spaces[orientation][y].hunger_fox = config->spaces[x][y].hunger_fox;

            // If the fox survive enough generations, it can reproduce once in a period of time (fox_gen)
            if (config->spaces[x][y].age_fox >= config->fox_gen)
                spaces[orientation][y].age_fox = -1;

            // If two or more foxes try to move to the same cell, only the younger is kept in the ecosystem
            else
                spaces[orientation][y].age_fox = MIN(spaces[orientation][y].age_fox, config->spaces[x][y].age_fox);
        }
        else if (type == RABBIT)
        {
            spaces[orientation][y].type = config->spaces[x][y].type;
            if (config->spaces[x][y].age_rabbit >= config->rabbit_gen)
                spaces[orientation][y].age_rabbit = -1;
            else
                spaces[orientation][y].age_rabbit = MIN(spaces[orientation][y].age_rabbit, config->spaces[x][y].age_rabbit);
        }
    }
    // Same operations but in the y coordinate (horizontal - columns)
    else
    {
        if (type == FOX)
        {
            spaces[x][orientation].type = config->spaces[x][y].type;
            spaces[x][orientation].hunger_fox = config->spaces[x][y].hunger_fox;
            if (config->spaces[x][y].age_fox >= config->fox_gen)
                spaces[x][orientation].age_fox = -1;
            else
                spaces[x][orientation].age_fox = MIN(spaces[x][orientation].age_fox, config->spaces[x][y].age_fox);
        }
        else if (type == RABBIT)
        {
            spaces[x][orientation].type = config->spaces[x][y].type;
            if (config->spaces[x][y].age_rabbit >= config->rabbit_gen)
                spaces[x][orientation].age_rabbit = -1;
            else
                spaces[x][orientation].age_rabbit = MIN(spaces[x][orientation].age_rabbit, config->spaces[x][y].age_rabbit);
        }
    }

    /*
    PS: When an iteration (generation of the ecosystem) passes and the age and hunger stats are set to -1,
    the evolve_system function call will increment all stats by 1 before the next iteration. Also, the
    comparisons are always done in the ecosystem source matrix, while the movements are all done in the
    auxiliary matrix to solve any conflicts and to don't interfere with the system's dynamics
    (all animals from each type move at the same time)
    */
    if (type == FOX)
    {
        // If the fox reproduces, it leaves a child in the cell it just left that has its age and hunger set to zero
        if (config->spaces[x][y].age_fox >= config->fox_gen)
        {
            // Foxes children don't inherit its hunger current state
            spaces[x][y].age_fox = -1;
            spaces[x][y].hunger_fox = -1;
            spaces[x][y].type = FOX;
        }
        // Otherwise, the place in the ecosystem is left empty with no animal stats
        else
        {
            spaces[x][y].type = EMPTY;
            spaces[x][y].age_fox = FOREVER;
            spaces[x][y].hunger_fox = FOREVER;
        }
    }
    else if (type == RABBIT)
    {
        if (config->spaces[x][y].age_rabbit >= config->rabbit_gen)
        {
            spaces[x][y].age_rabbit = -1;
            spaces[x][y].type = RABBIT;
        }
        else
        {
            spaces[x][y].type = EMPTY;
            spaces[x][y].age_rabbit = FOREVER;
        }
    }

}

void move_rabbits (environment *config, cell **spaces, unsigned int g)
{
    int x, y, i;
    int adj[4];
    int north, south, east, west;
    unsigned int available, index, next_cell;
    int chunk;

    #pragma omp parallel num_threads(N_THREADS) private(x, y, i, north, south, east, west, available, adj, index, next_cell)
    {
        chunk = config->r * config->c / N_THREADS;
        #pragma omp for schedule(static, chunk)
        for (x = 0; x < config->r; x++)
        {
            north = x - 1;
            south = x + 1;

            for (y = 0; y < config->c; y++)
            {
                for (i = 0; i < 4; i++)
                    adj[i] = -1;
                available = 0;
                east = y + 1;
                west = y - 1;

                // Enumerates the possible positions to move and stores them in the adj array (if any exists)
                if (config->spaces[x][y].type == RABBIT)
                    empty_cells(x, y, north, south, east, west, config, adj, &available, 0);

                // If there's any empty adjacent cell
                if (available > 0)
                {
                    // (G + X + Y) (mod P)
                    index = calculate_next_position(g, x, y, available);

                    // e.g., [EAST, WEST, -1, -1] ==> adj[1] = WEST
                    next_cell = adj[index];

                    #pragma omp critical
                    switch (next_cell)
                    {
                        case NORTH:
                            move_animal_to_adj(config, spaces, x, y, north, RABBIT, true);
                            break;
                        case EAST:
                            move_animal_to_adj(config, spaces, x, y, east, RABBIT, false);
                            break;
                        case SOUTH:
                            move_animal_to_adj(config, spaces, x, y, south, RABBIT, true);
                            break;
                        case WEST:
                            move_animal_to_adj(config, spaces, x, y, west, RABBIT, false);
                            break;
                        default:
                            break;
                    }
                }
            }
        }
    }
}

void move_foxes (environment *config, cell **spaces, unsigned int g)
{
    int eat[4], adj[4];
    int x, y, i;
    unsigned int available_to_eat, empty, index, next_cell;
    int north, south, east, west;
    int chunk = config->r * config->c / N_THREADS;

    #pragma omp parallel num_threads(N_THREADS) private(x, y, i, north, south, east, west, available_to_eat, eat, adj, empty, index, next_cell)
    {
        #pragma omp for schedule(static, chunk)
        for (x = 0; x < config->r; x++)
        {
            north = x - 1;
            south = x + 1;

            for (y = 0; y < config->c; y++)
            {
                for (i = 0; i < 4; i++)
                {
                    eat[i] = -1;
                    adj[i] = -1;
                }

                available_to_eat = 0;
                empty = 0;
                east = y + 1;
                west = y - 1;

                if (config->spaces[x][y].type == FOX)
                {
                    empty_cells(x, y, north, south, east, west, config, eat, &available_to_eat, RABBIT);

                    // If there isn't any rabbits nearby, then the fox tries to look for empty adjacent cells
                    if (eat[0] == -1)
                        empty_cells(x, y, north, south, east, west, config, adj, &empty, EMPTY);
                }

                // If there's at least one adjacent rabbit to eat
                if (available_to_eat > 0)
                {
                    index = calculate_next_position(g, x, y, available_to_eat);
                    next_cell = eat[index];

                    #pragma omp critical
                    switch (next_cell) {
                        case NORTH:
                            move_animal_to_adj(config, spaces, x, y, north, FOX, true);
                            // The fox hunger status is decreased to zero (just ate a rabbit)
                            spaces[north][y].hunger_fox = -1;
                            break;
                        case EAST:
                            move_animal_to_adj(config, spaces, x, y, east, FOX, false);
                            spaces[x][east].hunger_fox = -1;
                            break;
                        case SOUTH:
                            move_animal_to_adj(config, spaces, x, y, south, FOX, true);
                            spaces[south][y].hunger_fox = -1;
                            break;
                        case WEST:
                            move_animal_to_adj(config, spaces, x, y, west, FOX, false);
                            spaces[x][west].hunger_fox = -1;
                            break;
                        default:
                            break;
                    }
                }
                // Otherwise, if there's any empty adjacent cell to move
                else if (empty > 0)
                {
                    index = calculate_next_position(g, x, y, empty);
                    next_cell = adj[index];

                    #pragma omp critical
                    switch (next_cell)
                    {
                        case NORTH:
                            move_animal_to_adj(config, spaces, x, y, north, FOX, true);
                            break;
                        case EAST:
                            move_animal_to_adj(config, spaces, x, y, east, FOX, false);
                            break;
                        case SOUTH:
                            move_animal_to_adj(config, spaces, x, y, south, FOX, true);
                            break;
                        case WEST:
                            move_animal_to_adj(config, spaces, x, y, west, FOX, false);
                            break;
                        default:
                            break;
                    }
                }
            }
        }
    }
}

void copy_state (environment *config, cell **spaces)
{
    // Copies the ecosystem current state to the auxiliary matrix
    #pragma omp parallel for num_threads(N_THREADS)
    for (int i = 0; i < config->r; i++)
    {
        for (int j = 0; j < config->c; j++)
        {
            if (config->spaces[i][j].type == RABBIT)
                spaces[i][j].age_rabbit = config->spaces[i][j].age_rabbit;
            else if (config->spaces[i][j].type == FOX)
            {
                spaces[i][j].age_fox = config->spaces[i][j].age_fox;
                spaces[i][j].hunger_fox = config->spaces[i][j].hunger_fox;
            }
            else if (config->spaces[i][j].type == EMPTY || config->spaces[i][j].type == STONE)
            {
                spaces[i][j].age_rabbit = FOREVER;
                spaces[i][j].age_fox = FOREVER;
                spaces[i][j].hunger_fox = FOREVER;
            }
            spaces[i][j].type = config->spaces[i][j].type;
        }
    }
}

void update_state (environment *config, cell **spaces)
{
    // Updates the ecosystem state with the recent modifications done in the auxiliary matrix
    #pragma omp parallel for num_threads(N_THREADS)
    for (int i = 0; i < config->r; i++)
    {
        for (int j = 0; j < config->c; j++)
        {
            if (spaces[i][j].type == RABBIT)
            {
                config->spaces[i][j].age_rabbit = spaces[i][j].age_rabbit;
                config->spaces[i][j].hunger_fox = FOREVER;
            }
            else if (spaces[i][j].type == FOX)
            {
                config->spaces[i][j].age_fox = spaces[i][j].age_fox;
                config->spaces[i][j].hunger_fox = spaces[i][j].hunger_fox;
            }
            else if (spaces[i][j].type == STONE)
            {
                config->spaces[i][j].age_rabbit = FOREVER;
                config->spaces[i][j].age_fox = FOREVER;
                config->spaces[i][j].hunger_fox = FOREVER;
            }
            config->spaces[i][j].type = spaces[i][j].type;
        }
    }
}

void evolve_system (environment *config)
{
    // Ages all the animals in the ecosystem (and its hunger if it's a fox)
    #pragma omp parallel for num_threads(N_THREADS)
    for (int i = 0; i < config->r; i++)
    {
        for (int j = 0; j < config->c; j++)
        {
            if (config->spaces[i][j].type == RABBIT)
                config->spaces[i][j].age_rabbit++;
            else if (config->spaces[i][j].type == FOX)
            {
                config->spaces[i][j].age_fox++;
                config->spaces[i][j].hunger_fox++;

                // Eliminates foxes whose starved in the ecosystem
                if (config->spaces[i][j].hunger_fox >= config->fox_food)
                {
                    config->spaces[i][j].type = EMPTY;
                    config->spaces[i][j].age_rabbit = FOREVER;
                    config->spaces[i][j].age_fox = FOREVER;
                    config->spaces[i][j].hunger_fox = FOREVER;
                }
            }
        }
    }
}

void print_limits (unsigned int n)
{
    for (int i = 0; i < n; i++)
        printf("-");
    printf("\n");
}

void print_state (environment *config, unsigned int gen)
{
    int object;
    unsigned int header = 2 * config->c + 3;

    printf("\nGen %d\n", gen);
    print_limits(header);

    for (int x = 0; x < config->r; x++) {
        printf("| ");
        for (int y = 0; y < config->c; y++) {
            object = config->spaces[x][y].type;
            switch (object) {
                case STONE:
                    printf("* ");
                    break;
                case EMPTY:
                    printf("  ");
                    break;
                case FOX:
                    printf("R ");
                    break;
                case RABBIT:
                    printf("C ");
                    break;
                default:
                    break;
            }
        }
        printf("|\n");
    }
    print_limits(header);
}
