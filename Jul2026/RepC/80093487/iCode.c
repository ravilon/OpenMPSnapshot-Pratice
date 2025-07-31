#include <stdio.h>
#include <immintrin.h>
#include <omp.h>
#include <time.h>
#include <sys/time.h>

int main()
{ 
  //variable initialization...
  int iThreadCnt = 1, iLoopCntr = 1000*1000*1000;
  struct timeval start, end;
  float fTimeTaken = 0;
  int iCntr = 0;

  //initializing variables...
  register  __m256i vec1 = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
  register  __m256i vec2 = _mm256_set_epi32(9, 3, 6, 7, 9, 3, 6, 7);
  register  __m256i vec3 = _mm256_set_epi32(1, 1, 1, 1, 1, 1, 1, 1);
  register  __m256i vec4 = _mm256_set_epi32(4, 5, 3, 6, 4, 5, 1, 6);

  //start timer..
  gettimeofday(&start, NULL);
  
  #pragma omp parallel for default(shared)	                
  for (iCntr=0; iCntr < iLoopCntr; iCntr++)
  {
	if(iCntr == 0)
	{
		iThreadCnt = omp_get_num_threads();
	}
	__m256i result1 = _mm256_add_epi32(vec1, vec2);
	__m256i result2 = _mm256_add_epi32(vec3, vec4);
	__m256i result3 = _mm256_sub_epi32(result1, result2);
	__m256i result4 = _mm256_add_epi32(result1, result2);
	asm("");
  }		

  //end timer..
  gettimeofday(&end, NULL);
	
  fTimeTaken = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)); //time in micro-sec

  // 4 for the no. of operations performed inside the for loop...
  // iLoopCntr is the number of times the loop executes
  // iThreadCnt is the number of threads used to execute the code in parallel
  // 256/32 because we are dealing with integer values, and integer is 8 bytes in 64-bit machine
  // fTimeTaken is the time taken in micro-sec
  printf("Number of iops = %f per sec.\n", (4 * (float)iLoopCntr * (float)iThreadCnt * (256./32.) * 1000000) / fTimeTaken);
  
  return 0;
}
