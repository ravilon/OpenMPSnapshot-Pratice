#include <stdio.h>
#include <math.h>
#include <sys/time.h> 
#include <sys/resource.h>  
#include <omp.h>

double timeGetTime() 
{     
	struct timeval time;     
	struct timezone zone;     
	gettimeofday(&time, &zone);     
	return time.tv_sec + time.tv_usec*1e-6; 
}  


const long int VERYBIG = 100000;

int main( void )
{
  int i;
  long int j, k, sum;
  double sumx, sumy, total, z;
  double starttime, elapsedtime;
  int iterationNumber = 6;
  double times[iterationNumber];
  double allExperimentTime = 0;
  double averageTime;
  double threadTimes[4];
  // ---------------------------------------------------------------------
  // Output a start message
  printf( "OpenMP Parallel Timings for %ld iterations \n\n", VERYBIG );

  // repeat experiment several times
  for( i=0; i<iterationNumber; i++ )
  {
    // get starting time
    printf("Iteration %d:\n", i+1);
    starttime = timeGetTime();
    // reset check sum and total
    sum = 0;
    total = 0.0;

    // get time of each thread
    double wtime;
    
    // Work loop, do some work by looping VERYBIG times
    #pragma omp parallel num_threads(4) private( sumx, sumy, k, wtime ) shared(threadTimes)
    {
      wtime = omp_get_wtime();
      #pragma omp for reduction( +: sum, total ) schedule( dynamic, 2000 ) nowait
        for( int j=0; j<VERYBIG; j++ )
        {
          // increment check sum
          sum += 1;
         
          // Calculate first arithmetic series
          sumx = 0.0;
          for( k=0; k<j; k++ )
           sumx = sumx + (double)k;

          // Calculate second arithmetic series
          sumy = 0.0;
          for( k=j; k>0; k-- )
           sumy = sumy + (double)k;

          if( sumx > 0.0 )total = total + 1.0 / sqrt( sumx );
          if( sumy > 0.0 )total = total + 1.0 / sqrt( sumy );
        }
        wtime = omp_get_wtime() - wtime;
        int threadNumber = omp_get_thread_num();
        threadTimes[threadNumber] = wtime;
    }
    
    // get ending time and use it to determine elapsed time
    elapsedtime = timeGetTime() - starttime;
    times[i] = elapsedtime;
    allExperimentTime += elapsedtime;
    for (int i = 0; i < 4; i++)
      printf("\tThread %d Calculation Time: %.3f Seconds\n", i, threadTimes[i]); 
  
    // report elapsed time
    printf("\tTime Elapsed %10d mSecs Total=%lf Check Sum = %ld\n",
                   (int)(elapsedtime * 1000), total, sum );
  }
  averageTime = allExperimentTime / iterationNumber;
  printf("Average Execution Time: %.3f Seconds\n", averageTime);

  // return integer as required by function header
  return 0;
}
// **********************************************************************
