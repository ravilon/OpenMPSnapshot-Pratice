#pragma once
#include<stdio.h>
#include<string.h>
#include <stdlib.h>
#include<time.h>
#include "omp.h"

//macros
#define ALPHABET_LENGTH 4
#define max(x,y) ((x)>(y)?(x):(y))

int get_index_of_character(char *str,char x, int len);
void print_matrix(int **x, int row, int col);
void calc_P_matrix_v2(int **P, char *b, int len_b, char *c, int len_c);
int lcs_yang_v2(int *DP, int *prev_dp, int **P, char *A, char *B, char *C, int m, int n, int u);
int lcs(int **DP, char *A, char *B, int m, int n);