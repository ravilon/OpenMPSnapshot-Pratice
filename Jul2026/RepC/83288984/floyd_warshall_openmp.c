/* 
 * This file contains code that is the parallelized version of 
 * Floyd Warshall: All-Pair-Shortest-Path algorithm, written in C using 
 * OpenMP. Floyd Warshall algorithm basically finds all pair shortest 
 * paths in a weighted graph and use dynamic programming technique to 
 * solve the problem. 
 *
 * Compiling: via 1) GNU 
 *		  2) Oracle Solaris  
 * 
 *  1) gcc -o floyd_warshal_omp -fopenmp floyd_warshal_omp.c
 * 
 *          - or -
 * 
 *  2) cc -o floyd_warshal_omp -xopenmp floyd_warshal_omp.c 
 *
 * Running:
 * 
 *  ./floyd_warshal_omp [DATA_FILE_NAME] [NUMBER_OF_THREADS]
 * 
 *  Example:  ./floyd_warshal_omp test_data_1000.dat 4
 * 
 * 
 * File: floyd_warshal_omp.c                Author: Inteasar Haider
 * Date: January 01, 2017                   Version: V1.0
 * 
 */

#include <omp.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <sys/time.h>

/* Print matrix after initialization and calculating shortest path
 * 1 = Print matrix
 * 0 = Do not print matrix
 * 
 */ 
#define PRINT_MATRIX 0     
#define MIN_NODES 4	 /* Minimun nodes in graph		     */    
#define DATA_FOLDER  "../test_data/"

static void calculate_all_pair_shortest_path (void);
static void print_distance_adjacency_matrix (void);
static void calculate_execution_time (struct timeval time_start, 
        struct timeval time_end);
static int read_data_from_file (char data_file_name[]);

/* adjacency matrix to store distance between nodes of directed graph */
int **nodes_distance;
int num_nodes;

int main (int argc, char *argv[])
{
  int num_threads;		 /* threads to start upon execution */ 
  struct timeval time_start, time_end;  /* to hold start & end time */
  
  /* File name and number of threads to spawn will be passed via 
   * command line as CLI parameter 
   * First param (argv[1]) will be file name
   * Second param (argv[2]) will be number of threads to spawn
   */  
  if( argc == 3 ) {
    read_data_from_file (argv[1]);
    num_threads = atoi(argv[2]);
  } else {
    printf ("==> Error: Data file name (e.g: test_data_10.dat) as 1st "
	    "and number of threads as 2nd parameter is not passed "
	    "from command line\n\n");
    return EXIT_FAILURE;
  }

  printf ("==> Spawning %d threads to calculate shortest paths between"
          " %d nodes..\n\n", num_threads, num_nodes);
 
  print_distance_adjacency_matrix ();

  gettimeofday (&time_start, NULL);

  /* entering into the parallel region */
  #pragma omp parallel num_threads(num_threads) shared(nodes_distance)
  calculate_all_pair_shortest_path ();
  /* exiting from the parallel region */

  gettimeofday (&time_end, NULL);

  calculate_execution_time (time_start, time_end);
   
  print_distance_adjacency_matrix ();
  
  free(nodes_distance);
  return EXIT_SUCCESS;
}

/* This routine will be executed in parallel region and it's 
 * responsibility is to calculate shortest path and update shorter 
 * path distance in nodes_distance matrix. 
 * 
 * num_nodes will be evenly distributed among all threads and each 
 * thread will perform calculation simultaneously on given dataset to 
 * find the shortest distance between two nodes i and j via one 
 * intermediate node i.e K at a time and update, if found, shortest 
 * path in the nodes_distance matrix.
 *  
 * Once an iteration of an intermediate node i.e 'K' is done, all the 
 * threads will wait for others before starting the next iteration of 
 * intermediate node and that iteration will run (num_nodes-1) times. 
 * 
 * input parameters:	none
 * output parameters:	nodes_distance will have all shortest paths
 * return value:	none
 *
 */
void calculate_all_pair_shortest_path(void)
{
  int possible_short_dist; /* to hold distance between i and j via k */
  
  for (int k = 0; k < num_nodes; ++k)
  {
    /*
     * Wait for all threads to complete the iteration before we do 
     * shortest path calculations on the adjacency matrix for next 
     * intermediate node
     * 
     */
    # pragma omp barrier

    # pragma omp for
    for (int i = 0; i < num_nodes; ++i)
    {
      for (int j = 0; j < num_nodes; ++j)
      {
        /* if both nodes (i & j) are not same and path between node_i 
         * and Node_j via node_k exists (i.e. anything except 0)
         * 
         */
        if ( (nodes_distance[i][k] * nodes_distance[k][j] != 0)
                 && (i != j) ) 
        {
           possible_short_dist = nodes_distance[i][k] + 
                   nodes_distance[k][j];
          
          /* If path exists between i and j and that path distance 
           * is already lesser than newly calculated distance 
           * (via Node_k) do not update anything and continue with 
           * other possibilities
           * 
           */
          if ( (nodes_distance[i][j] <= possible_short_dist) &&
                  (nodes_distance[i][j] != 0) )
          {
            continue;
          }           

          /* Replace the current distance with new shortest distance */
          nodes_distance[i][j] = possible_short_dist;
        }
      }
    }
  }
}

/* Responsibility of this routine is to calculate final execution time 
 * of the program 
 * 
 * input parameters:	Two params of timeval struct
 *                      a) time_start: time when program entered in to 
 *                         parallel regions and threads were spawned
 *                      b) time_end: when all threads completed working
 * 
 * output parameters:	Printing time spent on the console
 * return value:	none
 *
 */
void calculate_execution_time (struct timeval time_start, 
                               struct timeval time_end) 
{  
  long long execution_time = 1000000LL
	* (time_end.tv_sec  - time_start.tv_sec)
	+ (time_end.tv_usec - time_start.tv_usec);

  double time_spent = (double) execution_time / 1000000;
  printf ("==> Finished calculating shortest path in %f seconds.\n\n", 
          time_spent );
}

/* Responsibility of this routine is to print two dimensional matrix
 * that contains the distance between the nodes of given graph
 * 
 * input parameters:	none
 * output:              distance matrix printed on the console
 * return value:	none
 * side effects:	could crash console if a huge matrix is being 
 *                      printed
 *
 */
void print_distance_adjacency_matrix (void) 
{
  if(PRINT_MATRIX==1)
  {
      int i, j;
      printf("    ");
      
      for (i = 0; i < num_nodes; ++i) 
      {
          printf("%4c", 'A' + i);
      }
      
      printf("\n");
      
      for (i = 0; i < num_nodes; ++i) 
      {
          printf("%4c", 'A' + i);
          for (j = 0; j < num_nodes; ++j)
          {
              printf("%4d", nodes_distance[i][j]);
          }
          printf("\n");
      }
      
      printf("\n");
  }
}

/* Responsibility of this routine is to populate the distance matrix 
 * with the test data being read from file
 * 
 * input parameters:	File name passed as a CLI Parameter
 * output:              distance matrix with test data
 * return value:	1 if unable to populate data
 *			0 if data is populated successfully 
 * side effects:	could crash program if test data file is 
 *			missing or contain in un-appropriate data
 *
 */
int read_data_from_file (char data_file_name[]) 
{
  FILE *data_file;
  int temp = 0;		 /* hold data read from file temporarily */
  int mem_size;          /* memory to allocate for adjacency matrix */ 

  char final_path[40];

  strcpy( final_path, DATA_FOLDER ); /* copy data folder path */
  strcat( final_path, data_file_name ); /* create file path */
  
  data_file = fopen (final_path, "r");
  
  if(data_file == NULL)                    
  {
    printf ("==> Cannot proceed, unable to find data file ..\n\n");
    exit(EXIT_FAILURE);
  }

  /* The first line will be the number of vertices */
  fscanf (data_file, "%d", &num_nodes);
  
  if(num_nodes <= MIN_NODES)	    /* graph should not be empty */
  {
    printf ("==> Cannot proceed, invalid graph size given ..\n\n");
    exit(EXIT_FAILURE);
  }
  
  /* Calculating memory needed for adjacency matrix */
  mem_size = (num_nodes * sizeof(int*)) +
	  (num_nodes * num_nodes * sizeof(int));

  /* Dynamically allocating memory for the adjacency matrix */
  nodes_distance = malloc (mem_size);

  /* Set the row indexes as pointers to the columns */
  for (int i = 0; i < num_nodes; ++i)
  {
    nodes_distance[i]  = (int*)(nodes_distance + num_nodes + 1) + 
	    (i * num_nodes);
  }
  
  printf("%d num_nodes being read from data file... \n", num_nodes);

  for (int i = 0; i < num_nodes; i++)
  {
    for (int j = 0; j < num_nodes; j++)
    {
      if (fscanf (data_file, "%d", &temp) == EOF)
      {
	break;
      } 
      else
      { 
        if (i == j) 
	{ 
	  /* distance between same node should be set zero */
	  nodes_distance[i][j] = 0;
	} 
	else 
	{
	  /* reading data from file */
	  nodes_distance[i][j]= temp;
	}
      }
    }
  }
  fclose (data_file);
  
  return EXIT_SUCCESS;
}
