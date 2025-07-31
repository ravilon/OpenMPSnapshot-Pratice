#include "par_grid_list.h"

int main(int argc, char* argv[]){

    char* file;                 /**< Input data file name */
    int generations = 0;        /**< Number of generations to proccess */
    int cube_size = 0;          /**< Size of the 3D space */
    
    GraphNode*** graph;         /**< Graph representation - 2D array of lists */
    List* update;               /**< Contains the information of nodes that might change state */

    /* Iterator variables */
    int g, i, j;
    GraphNode* g_it = NULL;
    Node* it = NULL;

    /* Lock variables */
    omp_lock_t list_lock;
    omp_lock_t** graph_lock;

    parseArgs(argc, argv, &file, &generations);

    /* Create an empty list, with size 0 */
    update = listCreate();

    graph = parseFile(file, update, &cube_size);
    
    /* Initialize lock variables */
    omp_init_lock(&list_lock);
    graph_lock = (omp_lock_t**)malloc(cube_size * sizeof(omp_lock_t*));
    for(i = 0; i < cube_size; i++){
        graph_lock[i] = (omp_lock_t*) malloc(cube_size * sizeof(omp_lock_t));
        for(j = 0; j < cube_size; j++){
            omp_init_lock(&(graph_lock[i][j]));
        }
    }
    double start = omp_get_wtime();  // Start Timer
    for(g = 1; g <= generations; g++){
        
        /* Convert list to vector */
        i = 0;
        int size = update->size;
        Node** vector = (Node**) malloc(sizeof(Node*) * size);
        for (it = listFirst(update); it != NULL; it = it->next){
            vector[i++] = it;
        }
        Node** proccessed;
        /* For each live node, inform its neighbors */
        #pragma omp parallel
        {
            #pragma omp for
            for (i = 0; i < size; i++){
                //printf("Neighbours processing by thread: %d\n", omp_get_thread_num());
                visitNeighbours(graph, graph_lock, cube_size, update, &list_lock, vector[i]->x, vector[i]->y, vector[i]->z);
            }
            
            /* Convert list to vector */
            #pragma omp single
            {
                i = 0;
                size = update->size;
                proccessed = (Node**) malloc(sizeof(Node*) * size);
                for (it = listFirst(update); it != NULL; it = it->next){
                    proccessed[i++] = it;
                }                
            }

            /* Update graph and update set */         
            #pragma omp for
            for (i = 0; i < size; i++){
                //printf("Update graph processing by thread: %d\n", omp_get_thread_num());
                Node* it = proccessed[i];
                unsigned char live_neighbours = it->ptr->neighbours;
                it->ptr->neighbours = 0;
                if(it->ptr->state == ALIVE){
                    if(live_neighbours < 2 || live_neighbours > 4){
                        graphNodeRemove(&(graph[it->x][it->y]), it->z, &(graph_lock[it->x][it->y]));
                        it->x = REMOVE;
                    }
                }else{
                    if(live_neighbours == 2 || live_neighbours == 3){
                        it->ptr->state = ALIVE; 
                    }
                    else{
                        graphNodeRemove(&(graph[it->x][it->y]), it->z, &(graph_lock[it->x][it->y]));
                        it->x = REMOVE;
                    }
                }
            }
        }

        /* Clean dead cells from the set */
        listCleanup(update);
        free(proccessed);
        free(vector);
    }

    double end = omp_get_wtime();   // Stop Timer
    
    /* Print the final set of live cells */
    printAndSortActive(graph, cube_size);
    time_print("%f\n", end - start);
    
    /* Free resources */
    freeGraph(graph, cube_size);
    listDelete(update);
    omp_destroy_lock(&list_lock);
    for(i = 0; i < cube_size; i++){
        for(j=0; j<cube_size; j++){
            omp_destroy_lock(&(graph_lock[i][j]));
        }
    }
    free(file);

    return 0;
}

/**************************************************************************/
void visitNeighbours(GraphNode*** graph, omp_lock_t** graph_lock, int cube_size,
                        List* list, omp_lock_t* list_lock,
                        coordinate x, coordinate y, coordinate z){

    GraphNode* ptr;
    coordinate x1, x2, y1, y2, z1, z2;
    x1 = (x+1)%cube_size; x2 = (x-1) < 0 ? (cube_size-1) : (x-1);
    y1 = (y+1)%cube_size; y2 = (y-1) < 0 ? (cube_size-1) : (y-1);
    z1 = (z+1)%cube_size; z2 = (z-1) < 0 ? (cube_size-1) : (z-1);
    /* If a cell is visited for the first time, add it to the update list, for fast access */
    if(graphNodeAddNeighbour(&(graph[x1][y]), z, &ptr, &graph_lock[x1][y])){ 
        listInsertLock(list, x1, y, z, ptr, list_lock);
    }
    if(graphNodeAddNeighbour(&(graph[x2][y]), z, &ptr, &graph_lock[x2][y])){ 
        listInsertLock(list, x2, y, z, ptr, list_lock);
    }
    if(graphNodeAddNeighbour(&(graph[x][y1]), z, &ptr, &graph_lock[x][y1])){ 
        listInsertLock(list, x, y1, z, ptr, list_lock);
    }
    if(graphNodeAddNeighbour(&(graph[x][y2]), z, &ptr, &graph_lock[x][y2])){ 
        listInsertLock(list, x, y2, z, ptr, list_lock);
    }
    if(graphNodeAddNeighbour(&(graph[x][y]), z1, &ptr, &graph_lock[x][y])){ 
        listInsertLock(list, x, y, z1, ptr, list_lock);
    }
    if(graphNodeAddNeighbour(&(graph[x][y]), z2, &ptr, &graph_lock[x][y])){ 
        listInsertLock(list, x, y, z2, ptr, list_lock);
    }
}

/**************************************************************************/
GraphNode*** initGraph(int size){

    int i,j;
    GraphNode*** graph = (GraphNode***) malloc(sizeof(GraphNode**) * size);

    for (i = 0; i < size; i++){
        graph[i] = (GraphNode**) malloc(sizeof(GraphNode*) * size);
        for (j = 0; j < size; j++){
            graph[i][j] = NULL;
        }
    }
    return graph;
}

/**************************************************************************/
void freeGraph(GraphNode*** graph, int size){

    int i, j;
    if (graph != NULL){
        for (i = 0; i < size; i++){
            for (j = 0; j < size; j++){
                graphNodeDelete(graph[i][j]);
            }
            free(graph[i]);
        }
        free(graph);
    }
}

/**************************************************************************/
void printAndSortActive(GraphNode*** graph, int cube_size){
    int x,y;
    GraphNode* it;
    for (x = 0; x < cube_size; ++x){
        for (y = 0; y < cube_size; ++y){
            /* Sort the list by ascending coordinate z */
            graphNodeSort(&(graph[x][y]));
            for (it = graph[x][y]; it != NULL; it = it->next){    
                /* At the end of each generation, the graph is guranteed to only have live cells */
                out_print("%d %d %d\n", x, y, it->z);
            }
        }
    }
}

/**************************************************************************/
void parseArgs(int argc, char* argv[], char** file, int* generations){
    if (argc == 3){
        char* file_name = malloc(sizeof(char) * (strlen(argv[1]) + 1));
        strcpy(file_name, argv[1]);
        *file = file_name;

        *generations = atoi(argv[2]);
        if (*generations > 0 && file_name != NULL)
            return;
    }    
    printf("Usage: %s [data_file.in] [number_generations]", argv[0]);
    exit(EXIT_FAILURE);
}

/**************************************************************************/
GraphNode*** parseFile(char* file, List* list, int* cube_size){
    
    int first = 0;
    char line[BUFFER_SIZE];
    int x, y, z;
    FILE* fp = fopen(file, "r");
    if(fp == NULL){
        err_print("Please input a valid file name");
        exit(EXIT_FAILURE);
    }
    GraphNode*** graph;

    while(fgets(line, sizeof(line), fp)){
        if(!first){
            if(sscanf(line, "%d\n", cube_size) == 1){
                first = 1;
                graph = initGraph(*cube_size);
            }    
        }else{
            if(sscanf(line, "%d %d %d\n", &x, &y, &z) == 3){
                /* Insert live nodes in the graph and the update set */
                graph[x][y] = graphNodeInsert(graph[x][y], z, ALIVE);
                listInsert(list, x, y, z, (GraphNode*) (graph[x][y]));                
            }
        }
    }

    fclose(fp);
    return graph;
}