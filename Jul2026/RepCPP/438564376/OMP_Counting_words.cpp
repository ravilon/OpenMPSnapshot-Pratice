#include <ctype.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]) { 
  int num_words = 0;
  int i=0;
// Parallel loop to check each char of the passed string  
#pragma omp parallel for
  for (i = 0; i <= strlen(argv[1]); i++) { 
#pragma omp atomic update
// In an atomic block, the number of words is increased by one,
// if the current char is a space and it is not followed by a space
// (counting successive as one space)
   if ((isspace(argv[1][i])) && !(isspace(argv[1][i+1])))
     num_words ++;
  }
// The number of total words in sentence is equal to 
// the number of seperators (space or successive spaces) 
// +1 (the last word after the last seperator)
  printf("Num words = %d\n", num_words+1);
}