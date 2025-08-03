/*
* In the following example of the sections construct the firstprivate clause is used to initialize the private copy of section_count of each thread. 
* The problem is  that the section constructs modify section_count, which breaks the independence of the section constructs. 

* When different threads execute each section, both sections will print the value 1. 
* When the same thread executes the two sections, one section will print the value 1 and the other will print the value 2.
* 
*/

#include <omp.h>
#include <stdio.h>
#define NT 4
int main( ) {
int section_count = 0;
omp_set_dynamic(0);
omp_set_num_threads(NT);
#pragma omp parallel
#pragma omp sections firstprivate( section_count )
{
#pragma omp section
{
section_count++;
/* may print the number one or two */
printf( "section_count %d\n", section_count );
}
#pragma omp section
{
section_count++;
/* may print the number one or two */
printf( "section_count %d\n", section_count );
}
}
return 1;
}

