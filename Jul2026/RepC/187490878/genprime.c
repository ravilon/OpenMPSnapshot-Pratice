#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int args, char *argv[]){
  //execution time
  double tstart = 0.0, ttaken;
  //check input correctness
  if (args != 3){
    printf("Invalid entry. Please enter one number between 2 and 100 000 and a thread counter between 0 and 100\n");
    exit(1);
  }
  int num = atoi(argv[1]);
  int threadCount = atoi(argv[2]);
  if(num <= 2 || num > 100000){
    printf("Entry too large. Please select a number between 2 and 100 000\n");
    exit(1);
  }
  if (threadCount > 100 || threadCount < 1){
	  printf("Entry invalid. Please enter a thread count between 1 abd 100\n");
	  exit(1);
  }

  // creating variables
  int numList[num];
  int i = 0;
  int floor = num / 2 +1;
  int primeBase = 2; // this is the first prime number
  int inner = 2; // this is where we first start
  int nextPrime = 2;
  numList[1] = 2; //init numList

  // Start counting time
  tstart = omp_get_wtime();

  // moving All for loops together so that we can resuse all the threads
  #pragma omp parallel num_threads(threadCount) \
				shared(numList, primeBase, num, floor) private(i, inner)
		while(primeBase < floor){
			#pragma omp for
			for(inner = primeBase; inner < num; inner++){
				// to avoid running into null, we use the indices to at least fill the void first
				if(primeBase == 2){
					if ((inner+1) % primeBase == 0) numList[inner] = 0;
          else numList[inner] = inner+1;
        }
        else{
					if ((numList[inner] != 0) && (numList[inner] % primeBase == 0)) numList[inner] = 0;
				}
			}// this is the end of the omp for loop

			//move on to next prime number
			#pragma omp single
      {
        for(nextPrime = primeBase; nextPrime < num; nextPrime++){
          if  (numList[nextPrime] != 0){
            primeBase = numList[nextPrime];
            // printf("Current prime: %d ", primeBase);
            break;
          }
        }
			}
		}

  // Finish calculating time
  ttaken = omp_get_wtime() - tstart;
  printf("Time take for the main part: %f\n",ttaken);

	// formating the answer
	int counter = 0;
	int len = sizeof(numList)/sizeof(int);
	for(i = 0; i < len; i++){
		if (numList[i] != 0) counter++;
	}
	int answer[counter];
	counter = 0;
	for(i = 0; i < len; i++){
		if (numList[i] != 0){
		answer[counter] = numList[i];
		counter++;
		}
	}

  // writing to the file
  FILE *fp;
  char output[100] = "";
  sprintf(output, "%d.sol",num);
  fp = fopen(output,"w");
  if (!fp){
    printf("Cannot create the file %s\n",output);
    exit(1);
  }

  fprintf(fp, "%d, %d, %d\n",1 ,2 ,0);
  for(i = 2; i <= counter; i++){
    int difference = 0;
    difference = answer[i-1] - answer[i-2];
    fprintf(fp, "%d, %d, %d\n",i, answer[i-1], difference);
  }
  fclose(fp);
  exit(0);
}
