#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include "hmm_data_gen.h"
#include "viterbi_helpers.h"

void viterbi( int n, int k, int t, int O[n], int S[k], float I[k], int Y[t], float A[k][k], float B[k][n]);

/*
 * Function: viterbi
 * --------------------
 *  Computes the most likely hidden state sequence based on a sequence of observations using Viterbi Algorithm
 *
 *  n: number of possible observations
 *  k: number of possible states
 *  t: length of observed sequence
 *  O: observation space
 *  S: state space
 *  I: prior probability - I[i] is the prior probability of S[i]
 *  Y: sequence of observations - Y[t] = i if observation at time t is O[i]
 *  A: transition probability - A[i,j] is the probability of going from state S[i] to S[j]
 *  B: emission probability - B[i,j] is the probability of observing O[j] given state S[i]
 *
 *  returns: the most likely hidden state sequence corresponding to given observations Y
 */
void viterbi(
  int n,
  int k,
  int t,
  int O[n],
  int S[k],
  float I[k],
  int Y[t],
  float A[k][k],
  float B[k][n]
) {
  // Initialize DP matrices
  float dp1[k][t]; // dp1[i,j] is the prob of most likely path of length j ending in S[i] resulting in the obs sequence
  int dp2[k][t]; // dp2[i,j] stores predecessor state of the most likely path of length j ending in S[i] resulting in the obs sequence
  for (int i=0; i<k; i++) {
    for (int j=0; j<t; j++) {
      if (j==0) {
        int observation = Y[0];
        dp1[i][0] = I[i]*B[i][observation]; // multiple init probability of state S[i] by the prob of observing init obs from state S[i]
        dp2[i][0] = 0;
      }
      else {
        dp1[i][j]=0.0;
        dp2[i][j]=0;
      }
    }
  }
  // Forward algorithm
  // Outer loop is iterating through t time stages
  for (int i=1; i<t; i++) {
    #pragma omp parallel
    {
    double start;
    double end;
    #pragma omp for
      // First inner loop is iterating through possible states //shared(dp1, dp2, A, B, Y, i, k)
      for (int j=0; j<k; j++) {
        float max = -1.0;
        int arg_max = -1;
        double curr_prob;
        // Second inner loop is iterating though possible states that could have preceded state S[j]
        for (int q=0; q<k; q++) {
          curr_prob = dp1[q][i-1] * A[q][j] * B[j][Y[i]];
          // Update max and curr_max if needed
          if (curr_prob > max) {
            max = curr_prob;
            arg_max = q;
          }
        }
        // Update dp memos
        dp1[j][i] = max;
        dp2[j][i] = arg_max;
      }
    }
  }

  // Get last state in path
  float max = dp1[0][t-1];
  int arg_max = 0;
  float state_prob;
  for (int i=1; i<k; i++) {
    state_prob = dp1[i][t-1];
    if (state_prob > max) {
      max = state_prob;
      arg_max = i;
    }
  }
  int X[t];
  X[t-1] = S[arg_max];
  // Backward algorithm
  for (int i=t-1; i>0; i--) {
    arg_max = dp2[arg_max][i];
    X[i-1] = S[arg_max];
  }
  printf("===========================================================\n"
         "RESULTS\n"
         "===========================================================\n");
  printf("Observation sequence:\n");
  print_arr(t,Y);
  printf("Most probable state sequence:\n");
  print_arr(t,X);
}

int main() {
  int n,k,t;
  printf("===========================================================\n"
         "VITERBI PARALLEL ALGORITHM\n");
  printf("Computing the most probable state sequence from a sequence of observations.\n");
  printf("This program will generate the observation sequence and HMM based on the\n"
          "dimensions you specify.\n");
  printf("===========================================================\n");
  printf("Enter the size of the observation space: ");
  scanf("%d",&n);
  printf("Enter the size of the state space: ");
  scanf("%d",&k);
  printf("Enter the number of observations in the sequence: ");
  scanf("%d",&t);
  int O[n];
  int S[k];
  int Y[t];
  float I[k];
  float A[k][k];
  float B[k][n];
  generate_sequence(k,n,t,O,S,Y,I,A,B);
  viterbi(n,k,t,O,S,I,Y,A,B);
  return 0;
}
