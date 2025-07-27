/****
 * Fix 1
 * This fix introduces padding between data
 * that we need to access in the array to
 * reduce false sharing.
 * NUMPAD represents the number of spaces
 * between addresses we need to access.
 *****/
 
#include "stdio.h"
#include <omp.h>
// #define NUMPAD 4
// #define NUMTHREADS 4

struct s{
float value;
int pad[4];
}Array[4];

int main( int argc, char *argv[ ] )
{
    const  int  SomeBigNumber = 100000000;  // keep less than 2B
    omp_set_num_threads(NUMTHREADS);
    
    //Start time
    double time0 = omp_get_wtime( );
    #pragma omp parallel for
    for( int i = 0; i < 4; i++ ){
        for( int j = 0; j < SomeBigNumber; j++ ){
            //Array[ i ].value  =  Array[ i ].value + (float)rand( );
            Array[ i ].value  =  Array[ i ].value + 2.;
        }
        
    }
    
    //Timing
    double time1 = omp_get_wtime( );
    double timeElapsed = time1- time0;
    
    //Performance
    
    //Speculative for now
    double megaAddsPerSecond = (double)SomeBigNumber*(double)4/timeElapsed/1000000.;
    
    //Write to a txt file
    FILE *fp;
    //headers
    fp = fopen("fix1.txt", "a");
    
    fprintf (fp, "%d\t", NUMPAD);
    fprintf (fp, "%d\t", NUMTHREADS);
    fprintf (fp, "%f\t",megaAddsPerSecond);
    fprintf (fp, "%f\t",1000000.*timeElapsed);
    fprintf (fp, "Fix 1");
    fprintf (fp, "\n");
    
    
    printf("Threads: %d\n", NUMTHREADS);
    printf("Padding: %d\n", NUMPAD);
    printf("Average Performance = %8.2lf MegaAdds/Sec\n", megaAddsPerSecond);
    printf("Elapsed Time=%10.2lf microseconds\n",1000000.*timeElapsed);
    printf ("Fix 1\n");
    printf("***** \n");
    
    return 0;
}

