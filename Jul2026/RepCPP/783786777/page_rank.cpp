#include "page_rank.h"

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>
#include <list>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//

void endLoop(Graph g, double *solution_curr, double *solution_prev, double convergence, int num_nodes, double& no_outgoing_sum, bool& is_converge){

  double sum0 = 0.0;
  double sum1 = 0.0;

  #pragma omp parallel for reduction(+:sum0, sum1)
  for (int i = 0; i < num_nodes; i++){
    sum0 += abs(solution_curr[i] - solution_prev[i]);
    if (outgoing_size(g, i) == 0)
      sum1 += solution_curr[i];
  }

  no_outgoing_sum = sum1;
  is_converge     = sum0 < convergence;
}

void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int     numNodes           = num_nodes(g);
  bool    is_converge        = false;
  double  equal_prob         = 1.0 / numNodes;
  double  one_minus_damping  = 1.0 - damping;
  double  no_outgoing_sum    = 0.;
  double* solution_prev      = new double[numNodes];
  double* reciprocal_outsize = new double[numNodes];

  #pragma omp parallel for reduction(+:no_outgoing_sum)
  for (int i = 0; i < numNodes; ++i)
  {
    solution[i] = equal_prob;
    solution_prev[i] = equal_prob;

    int out_size = outgoing_size(g, i);
    if (out_size == 0)
      no_outgoing_sum += equal_prob;
    else
      reciprocal_outsize[i] = 1.0 / out_size;
  }

  while (!is_converge) {

    memcpy(solution_prev, solution, numNodes * sizeof(double));

    #pragma omp parallel for
    for (Vertex vi = 0; vi < numNodes; ++vi){

      double sum = 0.;

      // sum over all nodes vj reachable from incoming edges
      for (const Vertex* vj_ptr = incoming_begin(g, vi); vj_ptr != incoming_end(g, vi); vj_ptr++)
        sum += solution_prev[*vj_ptr] * reciprocal_outsize[*vj_ptr];
      
      // (damping * score_new[vi]) + (1.0-damping) / numNodes
      sum = damping * sum + one_minus_damping * equal_prob;

      // sum over all nodes v in graph with no outgoing edges
      sum += damping * no_outgoing_sum * equal_prob;

      solution[vi] = sum;
    }

    endLoop(g, solution, solution_prev, convergence, numNodes, no_outgoing_sum, is_converge);

  }

  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */
}
