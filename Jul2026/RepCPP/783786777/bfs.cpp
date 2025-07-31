#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include <limits.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances)
{
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < frontier->count; i++)
    {

        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        {
            int outgoing = g->outgoing_edges[neighbor];
            if (distances[outgoing] != NOT_VISITED_MARKER)
                continue;
            if (__sync_bool_compare_and_swap(distances + outgoing, NOT_VISITED_MARKER, distances[node] + 1))
                new_frontier->vertices[__sync_fetch_and_add(&new_frontier->count, 1)] = outgoing;
        }
    }


    // #pragma omp parallel
    // {
    //     vertex_set thread_frontier;
    //     vertex_set_init(&thread_frontier, g->num_nodes);
        
    //     #pragma omp for
    //     for (int i = 0; i < frontier->count; i++)
    //     {

    //         int node = frontier->vertices[i];

    //         int start_edge = g->outgoing_starts[node];
    //         int end_edge = (node == g->num_nodes - 1)
    //                         ? g->num_edges
    //                         : g->outgoing_starts[node + 1];

    //         // attempt to add all neighbors to the new frontier
    //         for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
    //         {
    //             int outgoing = g->outgoing_edges[neighbor];
    //             if (distances[outgoing] == NOT_VISITED_MARKER)
    //                 thread_frontier.vertices[thread_frontier.count++] = outgoing;
    //         }
    //     }

    //     if (thread_frontier.count > 0)
    //     {
    //         int now_new_count = new_frontier->count;
    //         while (!__sync_bool_compare_and_swap(&new_frontier->count, now_new_count, thread_frontier.count + now_new_count))
    //         {
    //             now_new_count = new_frontier->count;
    //         }
    //         memcpy(
    //             new_frontier->vertices + now_new_count,
    //             thread_frontier.vertices, 
    //             thread_frontier.count * sizeof(int)
    //         );
    //     }

    //     free(thread_frontier.vertices); 
    // } 
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);
        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.8f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

void bottom_up_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances)
{   
    // #pragma omp parallel for schedule(dynamic, 512)
    // for (int i = 0; i < g->num_nodes; i++)
    // {
    //     if (distances[i] != NOT_VISITED_MARKER) continue;

    //     int  f_distance = distances[frontier->vertices[0]];
    //     bool is_connect = false;
    //     for (const int* in_ptr = incoming_begin(g, i); in_ptr < incoming_end(g, i); in_ptr++)
    //         if (distances[*in_ptr] == f_distance)
    //             is_connect = true;
        
    //     if (!is_connect) continue;

    //     distances[i] = f_distance + 1;
    //     new_frontier->vertices[__sync_fetch_and_add(&new_frontier->count, 1)] = i;
    // }

    #pragma omp parallel 
    {
        vertex_set thread_frontier;
        vertex_set_init(&thread_frontier, g->num_nodes);

        #pragma omp for schedule(dynamic, 512)
        for (int i = 0; i < g->num_nodes; i++)
        {
            if (distances[i] != NOT_VISITED_MARKER) continue;

            int  f_distance = distances[frontier->vertices[0]];
            bool is_connect = false;
            for (const int* in_ptr = incoming_begin(g, i); in_ptr < incoming_end(g, i); in_ptr++)
                if (distances[*in_ptr] == f_distance)
                    is_connect = true;
            
            if (!is_connect) continue;

            distances[i] = f_distance + 1;
            thread_frontier.vertices[thread_frontier.count++] = i;
        }

        if (thread_frontier.count > 0)
        {
            int now_new_count = new_frontier->count;
            while (!__sync_bool_compare_and_swap(&new_frontier->count, now_new_count, thread_frontier.count + now_new_count))
            {
                now_new_count = new_frontier->count;
            }
            memcpy(
                new_frontier->vertices + now_new_count,
                thread_frontier.vertices, 
                thread_frontier.count * sizeof(int)
            );
        }

        free(thread_frontier.vertices); 
    }
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {
        vertex_set_clear(new_frontier);
        bottom_up_step(graph, frontier, new_frontier, sol->distances);

        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

void hybrid_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances,
    int n_visited)
{
    if (n_visited < g->num_nodes / 2)
        top_down_step(g, frontier, new_frontier, distances);
    else
        bottom_up_step(g, frontier, new_frontier, distances);
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    int n_visited = 1;
    while (frontier->count != 0)
    {
        vertex_set_clear(new_frontier);
        hybrid_step(graph, frontier, new_frontier, sol->distances, n_visited);

        n_visited += new_frontier->count;

        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}
