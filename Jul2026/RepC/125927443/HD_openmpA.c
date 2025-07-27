/* C program to find hamming distance b/w
* two string
*Fine-Grained Code
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>


#define CHUNK_SIZE 1000
#define MASTER_THREAD 0

void PrintSomeMessage(char *p)
{
    printf("p: %s\n",p);
}

// driver code
int main(int argc,char **argv)
{
    //initialize sets' number and string's length
    int m,n,I,nthreads;
    if(argc<5)
    {
      printf("|Not enough arguments as input..\n");
      exit(EXIT_FAILURE);
    }
    else if(argc>5)
    {
      printf("|Too many arguments...\n");
      exit(EXIT_FAILURE);
    }
    else
    {
       m=atoi(argv[1]),n=atoi(argv[2]),I=atoi(argv[3]), nthreads=atoi(argv[4]);

      int i=0,j=0,TotalHammingDistance=0;

      //int ntreads=8;
      //creation of 2-dimentional matrices for setA and setB
      char **setA = malloc(m * sizeof(char *)); // Allocate row pointers
      for(i = 0; i < m; i++)
        setA[i] = malloc((I+1) * sizeof(char));  // Allocate each row separately

      char **setB = malloc(n * sizeof(char *)); // Allocate row pointers
      for(i = 0; i < n; i++)
        setB[i] = malloc((I+1) * sizeof(char));  // Allocate each row separately

      // initialize matrices with random string (0 and 1)
      for (i=0;i<m;i++){
          for(j=0;j<I;j++){
              setA[i][j]="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"[rand() % 62];
          }
          setA[i][I]='\0';
      }

      for (i=0;i<n;i++){
          for(j=0;j<I;j++){
              setB[i][j]="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"[rand() % 62];
          }
          setB[i][I]='\0';
      }
      printf("\n setA setB ready \n" );


      //creation of m*n matrix to store all hamming distances and initialize it
      int **HamDist = malloc(m * sizeof(int *)); // Allocate row pointers
      for(i = 0; i < m; i++)
        HamDist[i] = malloc(n * sizeof(int));

      printf("\n HamDist set \n" );

      int count=0;
      clock_t start=clock();
      // omp_set_nested(1);
      //
      // omp_set_dynamic(0);

      omp_set_num_threads(nthreads);
      // chunk=CHUNK_SIZE;
      #pragma omp parallel shared(setA,setB,HamDist) reduction(+:TotalHammingDistance)
      {
          int k=0,h=0;
          uint l=0;
          // tid = omp_get_thread_num();
    			// if (tid == MASTER_THREAD) {
    			// 	nthreads = omp_get_num_threads();
    			// 	printf("Number of threads = %d\n", nthreads);
    			// }
          //Calculate hamming distance for all combinations of the strings
        //  #pragma omp for schedule(dynamic, chunk) reduction(+:count,TotalHammingDistance)
          for (k=0;k<m;k++){
              for(h=0;h<n;h++){
                count=0;
                for(l=0;l<strlen(setB[h]);l++)
                {
                  if(setA[k][l]!=setB[h][l]) {
                    // #pragma omp critical
                    count++;

                  }
                }
                HamDist[k][h]=count;
                //printf("(%d)",HamDist[i][j]);
                // #pragma omp critical
                TotalHammingDistance+=HamDist[k][h];
              }
          }
      }
      clock_t end =clock();
      double hamm_time=(double)(end-start)/CLOCKS_PER_SEC;




      printf("\n|Total Hamming execution time= %f",hamm_time);
      printf("\n\n*|The Total Hamming Distance is: %d\n",TotalHammingDistance );
    }

    return 0;
}
