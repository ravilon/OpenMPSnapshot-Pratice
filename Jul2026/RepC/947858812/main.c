#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "function_declarations.h"

// ReSharper disable CppJoinDeclarationAndAssignment
// The wonderful world of MSVC. Because C99 support is too modern I suppose

// Descending order
int comp(const void* a, const void* b)
{
    const double diff = **(double**) b - **(double**) a;
    if (diff > 0) return 1;
    if (diff < 0) return -1;
    return 0;
}

void remove_argument(int* argc, char* argv[], const int n)
{
    // Check if the index is valid
    if (n < 1 || n >= *argc)
    {
        printf("Invalid index to remove.\n");
        return;
    }

    // Shift elements to the left
    for (int i = n; i < *argc - 1; i++)
    {
        argv[i] = argv[i + 1];
    }

    // Decrease the argument count
    (*argc)--;

    // Null-terminate the last element
    argv[*argc] = NULL;
}

int main(int argc, char** argv)
{
    printf("--- PageRank Serial ---\n");
    printf("Commandline arguments:\n");
    for (int i = 0; i < argc; ++i)
    {
        printf("argv[%d]: %s\n", i, argv[i]);
    }
    printf("\n");

    // Read arguments
    const char* small_graph_name = argv[1]; // "100nodes_graph.txt";
    const char* large_graph_name = argv[2]; // "web-stanford.txt";
    const double damping = atof(argv[3]);   // 0.85
    const double epsilon = atof(argv[4]);   // 0.0000001;
    const int top_pages = atoi(argv[5]);    // 10;

    // Concatenate strings to include full path
    const char* env_graph_path = getenv("GRAPH_DIR");
    const char* graphs_path = env_graph_path ? env_graph_path : "../../Graphs/";
    const size_t small_graph_path_length = strlen(graphs_path) + strlen(argv[1]) + 1;
    const size_t large_graph_path_length = strlen(graphs_path) + strlen(argv[2]) + 1;
    char* small_graph_path = malloc(small_graph_path_length);
    char* large_graph_path = malloc(large_graph_path_length);
    strcpy(small_graph_path, graphs_path);
    strcat(small_graph_path, small_graph_name);
    strcpy(large_graph_path, graphs_path);
    strcat(large_graph_path, large_graph_name);

    // --- Small graph ---
    printf("Small graph\n");
    int small_graph_nodes;
    double** small_graph;
    read_graph_from_file1(small_graph_path, &small_graph_nodes, &small_graph);
    printf("Number of pages: %d \n", small_graph_nodes);

    double* small_scores = malloc(small_graph_nodes * sizeof(double));
    PageRank_iterations1(small_graph_nodes, small_graph, damping, epsilon, small_scores);

    top_n_webpages(small_graph_nodes, small_scores, top_pages);

    printf("\n");

    // -- Large graph ---
    printf("Large graph\n");
    int large_graph_nodes;
    int* rows;
    int* cols;
    double* vals;
    read_graph_from_file2(large_graph_path, &large_graph_nodes, &rows, &cols, &vals);
    printf("Number of pages: %d \n", large_graph_nodes);

    double* large_scores = realloc(small_scores, large_graph_nodes * sizeof(double));
    if (!large_scores)
    {
        free(small_scores);
        exit(1);
    }

    PageRank_iterations2(large_graph_nodes, rows, cols, vals, damping, epsilon, large_scores);

    top_n_webpages(large_graph_nodes, large_scores, top_pages);

    free(small_graph_path);
    free(large_graph_path);
    free(rows);
    free(cols);
    free(vals);
    free(large_scores);

    // Remove the small graph from argv (shift other elements to the left)
    remove_argument(&argc, argv, 1);

    printf("\n");
    const int error = omp_pagerank(argc, argv);
    if (error != 0)
    {
        printf("OMP PageRank failed\n");
    }

    return 0;
}

void read_graph_from_file1(char* filename, int* N, double*** hyperlink_matrix)
{
    FILE* file = fopen(filename, "r");
    if (file == NULL)
    {
        printf("Error opening file %s\n", filename);
        exit(1);
    }

    // Read the file to find the number of nodes
    *N = 0;
    int max_buffer_size = 1024;
    char* line = malloc(max_buffer_size * sizeof(char));
    if (line == NULL)
    {
        printf("Memory allocation failed for line buffer\n");
        fclose(file);
        exit(1);
    }

    // Skip comment lines and find the line with "Nodes: X"
    while (fgets(line, max_buffer_size, file) != NULL)
    {
        // Check if the line was completely read (contains newline)
        if (strchr(line, '\n') == NULL && !feof(file))
        {
            // Line was too long for our buffer, resize and read the rest
            const int new_size = max_buffer_size * 2;
            char* new_line = realloc(line, new_size);
            if (new_line == NULL)
            {
                printf("Memory reallocation failed for line buffer\n");
                free(line);
                fclose(file);
                exit(1);
            }
            line = new_line;
            free(new_line);
            max_buffer_size = new_size;

            // Clear the remainder of the current line
            int c;
            while ((c = fgetc(file)) != '\n' && c != EOF) {}
            continue;
        }

        if (line[0] == '#')
        {
            // Check if this comment line contains the number of nodes
            if (strstr(line, "Nodes:") != NULL)
            {
                sscanf(line, "# Nodes: %d", N);
            }
        }
        else
        {
            // First non-comment line, break
            break;
        }
    }

    // If N wasn't found or is invalid, exit
    if (*N <= 0)
    {
        printf("Error: Could not determine the number of nodes\n");
        free(line);
        fclose(file);
        exit(1);
    }

    // Allocate memory for the hyperlink matrix
    *hyperlink_matrix = (double**) malloc(*N * sizeof(double*));
    if (*hyperlink_matrix == NULL)
    {
        printf("Memory allocation failed for hyperlink matrix rows\n");
        free(line);
        fclose(file);
        exit(1);
    }

    for (int i = 0; i < *N; i++)
    {
        (*hyperlink_matrix)[i] = (double*) calloc(*N, sizeof(double));
        if ((*hyperlink_matrix)[i] == NULL)
        {
            printf("Memory allocation failed for hyperlink matrix column %d\n", i);
            // Free previously allocated memory
            for (int j = 0; j < i; j++)
            {
                free((*hyperlink_matrix)[j]);
            }
            free(*hyperlink_matrix);
            free(line);
            fclose(file);
            exit(1);
        }
    }

    // Rewind the file to start reading the edges
    rewind(file);

    // Read the edges and fill the matrix
    int from_node, to_node;
    while (fgets(line, max_buffer_size, file) != NULL)
    {
        // Handle lines that are too long
        if (strchr(line, '\n') == NULL && !feof(file))
        {
            // Skip the rest of this line
            int c;
            while ((c = fgetc(file)) != '\n' && c != EOF) {}
            continue;
        }

        // Skip comment lines
        if (line[0] == '#') continue;

        // Parse the edge: FromNodeId ToNodeId
        if (sscanf(line, "%d %d", &from_node, &to_node) == 2)
        {
            // Make sure nodes are within range
            if (from_node >= 0 && from_node < *N && to_node >= 0 && to_node < *N)
            {
                // Set the link in the matrix to 1.0 initially
                (*hyperlink_matrix)[to_node][from_node] = 1.0;
            }
            else
            {
                printf("Warning: Node IDs out of range: %d -> %d\n", from_node, to_node);
            }
        }
    }

    // Normalize the matrix columns
    for (int j = 0; j < *N; j++)
    {
        double col_sum = 0.0;

        // Count incoming links
        for (int i = 0; i < *N; i++)
        {
            if ((*hyperlink_matrix)[i][j] > 0)
            {
                col_sum += 1.0;
            }
        }

        // Normalize if there are incoming links
        if (col_sum > 0)
        {
            for (int i = 0; i < *N; i++)
            {
                if ((*hyperlink_matrix)[i][j] > 0)
                {
                    (*hyperlink_matrix)[i][j] = 1.0 / col_sum;
                }
            }
        }
    }

    free(line);
    fclose(file);
}

void read_graph_from_file2(char* filename, int* N, int** row_ptr, int** col_idx, double** val)
{
    FILE* file = fopen(filename, "r");
    if (file == NULL)
    {
        printf("Error opening file %s\n", filename);
        exit(1);
    }

    // Read the file to find the number of nodes
    *N = 0;
    int max_buffer_size = 1024;
    // ReSharper disable once CppDFAMemoryLeak / idk what it's yapping about here - valgrind says no leaks
    char* line = malloc(max_buffer_size * sizeof(char));
    if (line == NULL)
    {
        printf("Memory allocation failed for line buffer\n");
        free(line);
        fclose(file);
        exit(1);
    }

    // Skip comment lines and find the line with "Nodes: X"
    while (fgets(line, max_buffer_size, file) != NULL)
    {
        // Check if the line was completely read (contains newline)
        if (strchr(line, '\n') == NULL && !feof(file))
        {
            // Line was too long for our buffer, resize and read the rest
            const int new_size = max_buffer_size * 2;
            char* new_line = realloc(line, new_size);
            if (new_line == NULL)
            {
                printf("Memory reallocation failed for line buffer\n");
                free(line);
                fclose(file);
                exit(1);
            }
            line = new_line;
            free(new_line);
            max_buffer_size = new_size;

            // Clear the remainder of the current line
            int c;
            while ((c = fgetc(file)) != '\n' && c != EOF) {}
            continue;
        }

        if (line[0] == '#')
        {
            // Check if this comment line contains the number of nodes
            if (strstr(line, "Nodes:") != NULL)
            {
                sscanf(line, "# Nodes: %d", N);
            }
        }
        else
        {
            // First non-comment line, break
            break;
        }
    }

    // If N wasn't found or is invalid, exit
    if (*N <= 0)
    {
        printf("Error: Could not determine the number of nodes\n");
        free(line);
        fclose(file);
        exit(1);
    }

    int* out_degree = calloc(*N, sizeof(int));
    int* in_edges_count = calloc(*N, sizeof(int));
    if (out_degree == NULL || in_edges_count == NULL)
    {
        printf("Memory allocation failed for out_degree or in_edges_count\n");
        free(out_degree);
        free(in_edges_count);
        free(line);
        fclose(file);
        exit(1);
    }

    // First pass to count out_degree and in_edges_count
    rewind(file);
    while (fgets(line, max_buffer_size, file) != NULL)
    {
        if (strchr(line, '\n') == NULL && !feof(file))
        {
            const int new_size = max_buffer_size * 2;
            char* new_line = realloc(line, new_size);
            if (new_line == NULL)
            {
                printf("Memory reallocation failed for line buffer\n");
                free(line);
                fclose(file);
                exit(1);
            }
            line = new_line;
            free(new_line);
            max_buffer_size = new_size;
            int c;
            while ((c = fgetc(file)) != '\n' && c != EOF) {}
            continue;
        }

        if (line[0] == '#') continue;

        int from_node, to_node;
        if (sscanf(line, "%d %d", &from_node, &to_node) == 2)
        {
            if (from_node >= 0 && from_node < *N && to_node >= 0 && to_node < *N)
            {
                out_degree[from_node]++;
                in_edges_count[to_node]++;
            }
        }
    }

    // Allocate in_edges
    int** in_edges = malloc(*N * sizeof(int*));
    if (in_edges == NULL)
    {
        printf("Memory allocation failed for in_edges\n");
        free(out_degree);
        free(in_edges_count);
        free(line);
        fclose(file);
        exit(1);
    }

    for (int i = 0; i < *N; i++)
    {
        if (in_edges_count[i] > 0)
        {
            in_edges[i] = malloc(in_edges_count[i] * sizeof(int));
            if (in_edges[i] == NULL)
            {
                printf("Memory allocation failed for in_edges[%d]\n", i);
                for (int j = 0; j < i; j++) free(in_edges[j]);
                free(in_edges);
                free(out_degree);
                free(in_edges_count);
                free(line);
                fclose(file);
                exit(1);
            }
        }
        else
        {
            in_edges[i] = NULL;
        }
    }

    int* current = calloc(*N, sizeof(int));
    if (current == NULL)
    {
        printf("Memory allocation failed for current array\n");
        for (int i = 0; i < *N; i++) free(in_edges[i]);
        free(in_edges);
        free(out_degree);
        free(in_edges_count);
        free(line);
        fclose(file);
        exit(1);
    }

    // Second pass to fill in_edges
    rewind(file);
    while (fgets(line, max_buffer_size, file) != NULL)
    {
        if (strchr(line, '\n') == NULL && !feof(file))
        {
            const int new_size = max_buffer_size * 2;
            char* new_line = realloc(line, new_size);
            if (new_line == NULL)
            {
                printf("Memory reallocation failed for line buffer\n");
                free(line);
                fclose(file);
                exit(1);
            }
            line = new_line;
            free(new_line);
            max_buffer_size = new_size;
            int c;
            while ((c = fgetc(file)) != '\n' && c != EOF) {}
            continue;
        }

        if (line[0] == '#') continue;

        int from_node, to_node;
        if (sscanf(line, "%d %d", &from_node, &to_node) == 2)
        {
            if (from_node >= 0 && from_node < *N && to_node >= 0 && to_node < *N)
            {
                in_edges[to_node][current[to_node]++] = from_node;
            }
        }
    }

    // Build CRS arrays
    *row_ptr = malloc((*N + 1) * sizeof(int));
    if (*row_ptr == NULL)
    {
        printf("Memory allocation failed for row_ptr\n");
        for (int i = 0; i < *N; i++) free(in_edges[i]);
        free(in_edges);
        free(out_degree);
        free(in_edges_count);
        free(current);
        free(line);
        fclose(file);
        exit(1);
    }

    (*row_ptr)[0] = 0;
    for (int i = 0; i < *N; i++)
    {
        (*row_ptr)[i + 1] = (*row_ptr)[i] + in_edges_count[i];
    }

    const int nnz = (*row_ptr)[*N];
    *col_idx = malloc(nnz * sizeof(int));
    *val = malloc(nnz * sizeof(double));
    if (*col_idx == NULL || *val == NULL)
    {
        printf("Memory allocation failed for col_idx or val\n");
        free(*row_ptr);
        for (int i = 0; i < *N; i++) free(in_edges[i]);
        free(in_edges);
        free(out_degree);
        free(in_edges_count);
        free(current);
        free(line);
        fclose(file);
        exit(1);
    }

    int k = 0;
    for (int i = 0; i < *N; i++)
    {
        for (int m = 0; m < in_edges_count[i]; m++)
        {
            int j = in_edges[i][m];
            (*col_idx)[k] = j;
            (*val)[k] = 1.0 / out_degree[j];
            k++;
        }
    }

    // Clean up
    for (int i = 0; i < *N; i++) free(in_edges[i]);
    free(in_edges);
    free(out_degree);
    free(in_edges_count);
    free(current);
    free(line);
    fclose(file);
}

void PageRank_iterations1(const int N, double** hyperlink_matrix, const double d, const double epsilon, double* scores)
{
    if (scores == NULL) return;

    // Initialize scores to 1/N
    for (int i = 0; i < N; ++i)
    {
        scores[i] = 1.0 / N;
    }

    // Find dangling nodes
    char dangling_flag = 0;
    char* dangling_indexes = calloc(N, sizeof(char));
    for (int i = 0; i < N; ++i)
    {
        double col_sum = 0.0;
        for (int j = 0; j < N; ++j)
        {
            col_sum += hyperlink_matrix[j][i];
        }
        if (col_sum == 0.0)
        {
            dangling_indexes[i] = 1;
            if (!dangling_flag) dangling_flag = 1;
        }
    }

    double diff = 100.0;
    double* old_scores = malloc(N * sizeof(double));

    while (diff > epsilon)
    {
        // Calculate sum of scores for dangling pages
        double dangling_scores = 0.0;
        if (dangling_flag)
        {
            for (int i = 0; i < N; ++i)
            {
                if (dangling_indexes[i] == 1)
                {
                    dangling_scores += scores[i];
                }
            }
        }

        // Copy of old scores for diff
        memcpy(old_scores, scores, N * sizeof(double));

        for (int i = 0; i < N; ++i)
        {
            scores[i] = (1 - d + d * dangling_scores) / N;
        }

        // Contribution from incoming edges
        for (int j = 0; j < N; ++j)
        {
            if (dangling_indexes[j] == 0) // Skip dangling nodes
            {
                for (int i = 0; i < N; ++i)
                {
                    if (hyperlink_matrix[i][j] > 0)
                    {
                        scores[i] += d * hyperlink_matrix[i][j] * old_scores[j];
                    }
                }
            }
        }

        // Calculate diff for convergence check
        diff = 0.0;
        for (int i = 0; i < N; ++i)
        {
            diff += fabs(scores[i] - old_scores[i]);
        }
    }

    free(dangling_indexes);
    free(old_scores);
}

void PageRank_iterations2(const int N, const int* row_ptr, const int* col_idx, const double* val, const double d,
                          const double epsilon, double* scores)
{
    if (scores == NULL) return;

    // Initialize scores to 1/N
    for (int i = 0; i < N; ++i)
    {
        scores[i] = 1.0 / N;
    }

    // Determine dangling nodes
    char* dangling_indexes = calloc(N, sizeof(char));

    // Initialize all as potentially dangling
    for (int j = 0; j < N; ++j)
    {
        dangling_indexes[j] = 1;
    }

    // Mark nodes that have outgoing edges
    int nnz = row_ptr[N];
    for (int k = 0; k < nnz; ++k)
    {
        int j = col_idx[k];
        dangling_indexes[j] = 0; // Not a dangling node
    }

    // Check if there are any dangling nodes
    char dangling_flag = 0;
    for (int j = 0; j < N; ++j)
    {
        if (dangling_indexes[j])
        {
            dangling_flag = 1;
            break;
        }
    }

    double* old_scores = malloc(N * sizeof(double));
    if (old_scores == NULL)
    {
        free(dangling_indexes);
        return;
    }

    double diff = 100.0;
    while (diff > epsilon)
    {
        // Calculate sum of scores from dangling pages
        double dangling_sum = 0.0;

        if (dangling_flag)
        {
            int j;
#pragma omp parallel for reduction(+:dangling_sum)
            for (j = 0; j < N; ++j)
            {
                if (dangling_indexes[j])
                {
                    dangling_sum += scores[j];
                }
            }
        }

        // Save current scores to old_scores
        memcpy(old_scores, scores, N * sizeof(double));

        // Set new scores base value
        const double base = (1.0 - d + d * dangling_sum) / N;
        for (int i = 0; i < N; ++i)
        {
            scores[i] = base;
        }

        // Add contributions from incoming edges
#ifdef _MSC_VER // Disgusting MSVC version
        int i;
#pragma omp parallel for
        for (i = 0; i < N; ++i) {
            const int start = row_ptr[i];
            const int end = row_ptr[i + 1];
            for (int k = start; k < end; ++k)
            {
                int j = col_idx[k]; // j is the source node with a link to i
                double contribution = d * val[k] * old_scores[j];
#pragma omp atomic
                scores[i] += contribution;
            }
        }
#else // Based gcc/clang version
#pragma omp parallel for reduction(+:scores[:N])
        for (int i = 0; i < N; ++i)
        {
            const int start = row_ptr[i];
            const int end = row_ptr[i + 1];
            for (int k = start; k < end; ++k)
            {
                int j = col_idx[k]; // j is the source node with a link to i
                scores[i] += d * val[k] * old_scores[j];
            }
        }
#endif

        // Compute the difference between new and old scores
        diff = 0.0;
        int j;
#pragma omp parallel for reduction(+:diff)
        for (j = 0; j < N; ++j)
        {
            diff += fabs(scores[j] - old_scores[j]);
        }
    }

    free(dangling_indexes);
    free(old_scores);
}

/**
 * List the top n webpages
 *
 * @param N total number of webpages (length of scores)
 * @param scores array of PageRank scores
 * @param n how many webpages to list
*/
void top_n_webpages(int N, double* scores, int n)
{
    if (scores == NULL) return;

    if (n > N) n = N;

    // Sort array of pointers to scores, to easily be able to retrieve original index
    double** score_pointers = malloc(N * sizeof(double*));
    int i;
#pragma omp parallel for
    for (i = 0; i < N; ++i)
    {
        score_pointers[i] = &scores[i];
    }
    qsort(score_pointers, N, sizeof(double), comp);

    printf("Top %d pages, sorted according to PageRank score:\n", n);
    for (int j = 0; j < n; ++j)
    {
        printf("Score [%lf] Index [%zu]\n", *score_pointers[j], (size_t) (score_pointers[j] - scores));
    }

    free(score_pointers);
}
