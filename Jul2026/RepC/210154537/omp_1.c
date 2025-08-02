#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <sys/time.h>

/*#define n 100 
#define m 100		
#define l 100 
#define NUM_THREADS 4*/


void initialize_string(char *str,int l) ;
double gettime(void) ;


int main()
{


		int i, j,r ,m=0,n=0,l=0,NUM_THREADS=0;
	int chunk, nthreads, tid;

scanf("%d",&m);
   
    scanf("%d",&n);
 
    scanf("%d",&l);
 scanf("%d",&NUM_THREADS);

int **hamming_array =(int**) malloc((m * sizeof (int*)));
 char **A = malloc(sizeof(char*)*m);
  char   **B = malloc(sizeof(char*)*n);
/////////////////////////////// MALLOC ///////////////////////////////////
	 for ( i=0; i<m;i++ )
   	 {
	    A[i]= malloc((l+1)*sizeof(char));
	    initialize_string(A[i],l);
	}

	//Create and initialize B [ #strings]   	
	for ( i=0; i<n;i++ )
	{
	    B[i]= malloc((l+1)*sizeof(char));
	    initialize_string(B[i],l);
	  
	}

	for (int q=0; q<m; q++)
         hamming_array[q] = (int *)malloc(n * sizeof(int));
/////////////////////////////////////////////////////////////////////////////


long long  sum_hamming=0;
int hamming_counter=0;
	chunk = (int)l/NUM_THREADS;
	
		
		omp_set_num_threads(NUM_THREADS);

	double time0 = gettime();	
		#pragma omp parallel shared(A, B , hamming_array, nthreads, chunk) private(i, j, r)
		{
		
			
			#pragma omp for schedule(static, chunk) collapse(3) 
			for (i=0; i<n; i++) {
				for (j=0; j<m; j++) {
					for (r=0; r<l; r++) {
 						 if(A[i][r]!=B[j][r]) 			
							hamming_array[i][j]+=1;

					}
				}
			}
		}

	
	
	double time1 = gettime();	
   
    for(i=0; i<m; i++) 
    {
	
      	for(int j=0;j<n;j++) 
      	{
	
		 	sum_hamming=sum_hamming+hamming_array[i][j];
	
		}
	}

	

	printf("---------------------------- \n");
	printf("-**************************- \n");
	printf("Sum of Hamming Distance = %lli \n",sum_hamming);
	printf("---------------------------- \n"); 

	double elapsed_time = time1-time0;

	printf("t0 = %lf and t1 = %lf \n",time0,time1);
	printf("Program time execution = %lf \n",elapsed_time);

	return 0;
}
double gettime(void)
{

	struct timeval ttime;
	gettimeofday(&ttime , NULL);
	return ttime.tv_sec + ttime.tv_usec * 0.000001;

}

void initialize_string(char *str,int l) 
{
	
    static const char alphanum[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890";

    for (int j = 0; j < l; ++j) 
    {
        str[j] = alphanum[rand() % (sizeof(alphanum) - 1)];
    }

    //str[l+1] ='0';

}


