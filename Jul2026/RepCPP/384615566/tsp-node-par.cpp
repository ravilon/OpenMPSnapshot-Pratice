#include <iostream>
#include <vector>
#include <queue>
#include <utility>
#include <cstring>
#include <climits>
#include <omp.h>

#include "../../tests/readfiles.h"

using namespace std;

// Sentinel value for representing `INFINITY`
#define INF INT_MAX

int result_tsp[N + 1];
int global_cost;

// Function to find the minimum edge cost
// having an end at the vertex i
int firstMin(int adj[N][N], int i)
{
    int min = INT_MAX;
    for (int k = 0; k < N; k++)
        if (adj[i][k] < min && i != k)
            min = adj[i][k];
    return min;
}

// function to find the second minimum edge cost
// having an end at the vertex i
int secondMin(int adj[N][N], int i)
{
    int first = INT_MAX, second = INT_MAX;
    for (int j = 0; j < N; j++)
    {
        if (i == j)
            continue;

        if (adj[i][j] <= first)
        {
            second = first;
            first = adj[i][j];
        }
        else if (adj[i][j] <= second &&
                 adj[i][j] != first)
            second = adj[i][j];
    }
    return second;
}

// State Space Tree nodes
struct Node
{
    // stores edges of the state-space tree
    // help in tracing the path when the answer is found
    vector<pair<int, int>> path;

    // stores the reduced matrix
    int reducedMatrix[N][N];

    // stores the lower bound
    int cost;

    // stores the current city number
    int vertex;

    // stores the total number of cities visited so far
    int level;
};

// Function to allocate a new node `(i, j)` corresponds to visiting city `j`
// from city `i`
Node *newNode(int parentMatrix[N][N], vector<pair<int, int>> const &path,
              int level, int i, int j)
{
    Node *node = new Node;

    // stores ancestors edges of the state-space tree
    node->path = path;
    // skip for the root node
    if (level != 0)
    {
        // add a current edge to the path
        node->path.push_back(make_pair(i, j));
    }

    // copy data from the parent node to the current node
    memcpy(node->reducedMatrix, parentMatrix,
           sizeof node->reducedMatrix);

    // Change all entries of row `i` and column `j` to `INFINITY`
    // skip for the root node
    for (int k = 0; level != 0 && k < N; k++)
    {
        // set outgoing edges for the city `i` to `INFINITY`
        node->reducedMatrix[i][k] = INF;

        // set incoming edges to city `j` to `INFINITY`
        node->reducedMatrix[k][j] = INF;
    }

    // Set `(j, 0)` to `INFINITY`
    // here start node is 0
    node->reducedMatrix[j][0] = INF;

    // set number of cities visited so far
    node->level = level;

    // assign current city number
    node->vertex = j;

    // return node
    return node;
}

// Function to reduce each row so that there must be at least one zero in each row
void rowReduction(int reducedMatrix[N][N], int row[N])
{
    // initialize row array to `INFINITY`
    fill_n(row, N, INF);

    // `row[i]` contains minimum in row `i`
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (reducedMatrix[i][j] < row[i])
            {
                row[i] = reducedMatrix[i][j];
            }
        }
    }

    // reduce the minimum value from each element in each row
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (reducedMatrix[i][j] != INF && row[i] != INF)
            {
                reducedMatrix[i][j] -= row[i];
            }
        }
    }
}

// Function to reduce each column so that there must be at least one zero
// in each column
void columnReduction(int reducedMatrix[N][N], int col[N])
{
    // initialize all elements of array `col` with `INFINITY`
    fill_n(col, N, INF);

    // `col[j]` contains minimum in col `j`
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (reducedMatrix[i][j] < col[j])
            {
                col[j] = reducedMatrix[i][j];
            }
        }
    }

    // reduce the minimum value from each element in each column
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (reducedMatrix[i][j] != INF && col[j] != INF)
            {
                reducedMatrix[i][j] -= col[j];
            }
        }
    }
}

// Function to get the lower bound on the path starting at the current minimum node
int calculateCost(int reducedMatrix[N][N])
{
    // initialize cost to 0
    int cost = 0;

    // Row Reduction
    int row[N];
    rowReduction(reducedMatrix, row);

    // Column Reduction
    int col[N];
    columnReduction(reducedMatrix, col);

    // the total expected cost is the sum of all reductions
    for (int i = 0; i < N; i++)
    {
        cost += (row[i] != INT_MAX) ? row[i] : 0,
                cost += (col[i] != INT_MAX) ? col[i] : 0;
    }

    return cost;
}

// Function to print list of cities visited following least cost
void printPath(vector<pair<int, int>> const &list)
{
    std::cout << "Path is: ";
    int i = 0;
    for (i = 0; i < list.size(); i++)
    {
        // cout << list[i].first + 1 << " —> " << list[i].second + 1 << endl;
        std::cout << list[i].first + 1 << " ";
        result_tsp[i] = list[i].first + 1;
    }
    std::cout << list[i - 1].second + 1 << endl;
    result_tsp[i] = list[i - 1].second + 1;
}

// Comparison object to be used to order the heap
struct comp
{
    bool operator()(const Node *lhs, const Node *rhs) const
    {
        return lhs->cost > rhs->cost;
    }
};

priority_queue<Node *, vector<Node *>, comp> global_pq;

// Function to solve the traveling salesman problem using Branch and Bound
void solve(int costMatrix[N][N])
{
    // Create a priority queue to store live nodes of the search tree
    priority_queue<Node *, vector<Node *>, comp> my_pq;

    vector<pair<int, int>> v;

    // create a root node and calculate its cost.
    // The TSP starts from the first city, i.e., node 0
    Node *root = newNode(costMatrix, v, 0, -1, 0);

    // get the lower bound of the path starting at node 0
    root->cost = calculateCost(root->reducedMatrix);

    // Add root to the list of live nodes
    // pq.push(root);

    // Initialize thread-local best tour
#pragma omp parallel private(my_pq)
    {
#pragma omp for
        for (int j = 1; j < N; j++)
        {
            // create a child node and calculate its cost
            Node *local_node = newNode(root->reducedMatrix, v, 1, 0, j);

            /* Cost of the child =
                cost of the parent node +
                cost of the edge(i, j) +
                lower bound of the path starting at node j
            */

            local_node->cost = root->cost + root->reducedMatrix[0][j] + calculateCost(local_node->reducedMatrix);
            my_pq.push(local_node);
        }

        // Finds a live node with the least cost, adds its children to the list of
        // live nodes, and finally deletes it from the list
        while (!my_pq.empty())
        {
            // Find a live node with the least estimated cost
            Node *min = my_pq.top();

            // The found node is deleted from the list of live nodes
            my_pq.pop();

            int i = 0;

            // `i` stores the current city number
            i = min->vertex;

            // if all cities are visited
            if (min->level == N - 1)
            {
#pragma omp critical
                {
                    /* copy all node struct to another, beacuse is deleted */
                    Node *node_tmp = new Node;
                    node_tmp->path = min->path;
                    memcpy(node_tmp->reducedMatrix, min->reducedMatrix, sizeof min->reducedMatrix);
                    node_tmp->cost = min->cost;
                    node_tmp->vertex = min->vertex;
                    node_tmp->level = min->level;

                    global_pq.push(node_tmp);
                }
                // // return to starting city
                // min->path.push_back(make_pair(i, 0));

                // // print list of cities visited
                // printPath(min->path);

                // // return optimal cost
                // global_cost = min->cost;

            }

            // do for each child of min
            // `(i, j)` forms an edge in a space tree

            for (int j = 0; j < N; j++)
            {
                // printf("Thread: %d\n", omp_get_thread_num());
                if (min->reducedMatrix[i][j] != INF)
                {
                    // create a child node and calculate its cost
                    Node *child = newNode(min->reducedMatrix, min->path, min->level + 1, i, j);

                    /* Cost of the child =
                        cost of the parent node +
                        cost of the edge(i, j) +
                        lower bound of the path starting at node j
                    */

                    child->cost = min->cost + min->reducedMatrix[i][j] + calculateCost(child->reducedMatrix);

                    // Add a child to the list of live nodes
                    my_pq.push(child);
                }
            }

            // free node as we have already stored edges `(i, j)` in vector.
            // So no need for a parent node while printing the solution.
            delete min;
        }
    }

}

int main(int argc, char *argv[])
{
    int n;
    int costMatrix[N][N];

    /* Read Files and Set variables */
    int result_to_compare[N + 1];
    int result_cost_to_compare;
    read_matrix_and_result_from_file(argv[1], &n, &result_cost_to_compare, costMatrix, result_to_compare);

    int thread_count = strtol(argv[2], NULL, 10);
    // initialize number of threads for paralel programming with openmp
    omp_set_num_threads(thread_count);

    double start = omp_get_wtime();
    solve(costMatrix);
    double stop = omp_get_wtime();

    // Find a live node with the least estimated cost
    Node *best_path = global_pq.top();

    // The found node is deleted from the list of live nodes
    global_pq.pop();

    // return to starting city
    // min->path.push_back(make_pair(i, 0));

    // print list of cities visited
    printPath(best_path->path);

    cout << "Best tour cost is: " << best_path->cost << endl;

    // Print time taken
    double time = stop - start;
    print_time(time, true);

    // Test if the result is correct
    test(result_tsp, result_to_compare, best_path->cost, result_cost_to_compare);

    return 0;
    // g++ -std=c++17 -Xpreprocessor -fopenmp tsp-node-par.cpp -lomp
}