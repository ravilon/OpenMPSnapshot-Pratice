/*
* The following example shows the use of the final clause and the 
* omp_in_final API call in a recursive binary search program. To reduce overhead once a * certain depth of recursion is reached the program uses the final clause to create 
* included tasks - which alloq additional optimisations.
*
* The use of the omp_in_final API call allows programmers to optimize their code by 
*  specifying which parts of the program are not necessary when a task can create only 
*  included tasks (that is, the code is inside a final task).
*
*  In this example, the use of a different state variable is not necessary so once the program reaches the part of the 
*   computation that is finalized and copying from the parent state to the new state is eliminated.
*
*   The final clause is most effective when used in conjunction with the mergeable 
*   clause since all tasks created in a final task region are included tasks that 
*   can be merged if the mergeable clause is present.
*/

#include<stdlib.h>
#include<stdio.h>
#include <string.h>
#include <omp.h>
#define LIMIT  3 /* arbitrary limit on recursion depth */
void check_solution(int *state);
void bin_search (int pos, int n, int *state)
{
if ( pos == n ) {
// check_solution(state);
return;
}
#pragma omp task final( pos > LIMIT ) mergeable
{
int new_state[n];
if (!omp_in_final() ) {
memcpy(new_state, state, pos );
state = new_state;
}
state[pos] = 0;
bin_search(pos+1, n, state );
}
#pragma omp task final( pos > LIMIT ) mergeable
{
int new_state[n];
if (! omp_in_final() ) {
memcpy(new_state, state, pos );
state = new_state;
}
state[pos] = 1;
bin_search(pos+1, n, state );
}
#pragma omp taskwait
}

int main()
{
int arr[] = {2, 3, 4, 10, 40}; 
int n = sizeof(arr)/ sizeof(arr[0]); 
int x = 10; 
bin_search(0, n, arr);
return 0;
}


















