/*
Threaded Programing - Assignment 2
Exam number B137122
*/

#include <stdio.h>
#include <math.h>


#define N 729
#define reps 1000
#include <omp.h>

double a[N][N], b[N][N], c[N];
int jmax[N];


void init1(void);
void init2(void);
void runloop(int);
void loop1chunk(int, int);
void loop2chunk(int, int);
void valid1(void);
void valid2(void);


int main(int argc, char *argv[]) {

  double start1,start2,end1,end2;
  int r;

  init1();

  start1 = omp_get_wtime();

  for (r=0; r<reps; r++){
    runloop(1);
  }

  end1  = omp_get_wtime();

  valid1();

  printf("Total time for %d reps of loop 1 = %f\n",reps, (float)(end1-start1));


  init2();

  start2 = omp_get_wtime();

  for (r=0; r<reps; r++){
    runloop(2);
  }

  end2  = omp_get_wtime();

  valid2();

  printf("Total time for %d reps of loop 2 = %f\n",reps, (float)(end2-start2));

}

void init1(void){
  int i,j;

  for (i=0; i<N; i++){
    for (j=0; j<N; j++){
      a[i][j] = 0.0;
      b[i][j] = 3.142*(i+j);
    }
  }

}

void init2(void){
  int i,j, expr;

  for (i=0; i<N; i++){
    expr =  i%( 3*(i/30) + 1);
    if ( expr == 0) {
      jmax[i] = N;
    }
    else {
      jmax[i] = 1;
    }
    c[i] = 0.0;
  }

  for (i=0; i<N; i++){
    for (j=0; j<N; j++){
      b[i][j] = (double) (i*j+1) / (double) (N*N);
    }
  }

}


void runloop(int loopid)  {

  // --------------------------------
  // declaration of shared variables
  // --------------------------------
  int nthreads = omp_get_max_threads();               // number of threads
  int limits_lo[nthreads];									          // array holding not computed low limits
  int limits_hi[nthreads];									          // array holding not computed high limits
  int ipt = (int) ceil((double)N/(double)nthreads);   // n/p iterations
  float ipr = 1.0/nthreads;                           // 1/p ratio

  // ****************************
  // starting paralell region
  //  at this point the code is forked to the threads and all the variables decleared inside tihs region are private
  // ****************************
  #pragma omp parallel default(none) shared(loopid, nthreads, limits_lo, limits_hi, ipt, ipr)
  {
    int myid  = omp_get_thread_num();                 // threads id
    limits_lo[myid] = myid*ipt;                       // set low limits for every thread
    limits_hi[myid] = (myid+1)*ipt;                   // set high limits for every thread
    if (limits_hi[myid] > N) limits_hi[myid] = N;     // case if limit is higher than N

    int lo = 0, hi = 0;
    int thrd = 0;                                     // auxiliary variable for stealing

    // ****************************
    // synchronise all threads
    //  we need to synchronise the threads so all of them start at the same time entering the while loop.
    //  this barrier is necessary as it is possible that one thread accesses the while loop completes its local set
    //  and steals another threads chunk before the other thread even enters the while loop.
    // ****************************
    #pragma omp barrier

    // --------------------------------
    // start while loop
    //  loop until ther is no more work to be done
    // --------------------------------
		while(thrd != -1) {

      // ****************************
      // starting critical region
      //  we check if a given thread still has iterations left to compute and update $lo and $hi that are later used for calling
      //  loop1 and loop2. the update of $hi and $lo requires access and update to the shared arrays $limits_lo and $limits_hi.
      //  in order to ensure that multiple threads cannot write to the shared array we need to put this in a critical region.
      // ****************************
      #pragma omp critical
      {
        thrd = -1;

        // ------------------------
        // set tread id
        //  in this section we check if a given thread still has iterations left to compute and set the thread id
        // ------------------------
      	if (limits_hi[myid] - limits_lo[myid] > 0) {   // check if iterations in local set are avalible
      		thrd = myid;
      	}
        // find most loaded thread
      	else {                                         // no iterations availible for thread
      		int thrd_iter = -1;                          // “most loaded” thread id
          int temp_iter = 0;                           // temprarly variable to find highest iteration count
          // find “most loaded” thread
      		for (int i = 0; i < nthreads; i++) {
      			int limits_d = limits_hi[i] - limits_lo[i];
      			if (limits_d > temp_iter) {
      				temp_iter = limits_d;
              thrd_iter = i;
      			}
      		}
          thrd = thrd_iter;
      	}

        // ------------------------
        // update hi and lo
        //  in this section we are updating hi and lo
        // ------------------------
        if (thrd != -1) {
        	int iter_remain = limits_hi[thrd] - limits_lo[thrd]; // calc remaining iterations
        	int remain = (int) ceil(ipr * iter_remain);
          // assign chunk_remain to free thread
        	lo = limits_lo[thrd];
        	hi = limits_lo[thrd] + remain;
        	limits_lo[thrd] = limits_lo[thrd] + remain;    // shared variable limits_lo is updated so that the other threads can access the new information
        }
      }

      // ------------------------
      // call loop1 or loop2
      //  call and execution of loop1 and loop2
      // ------------------------
      if (thrd != -1) {
        switch (loopid) {
           case 1: loop1chunk(lo,hi); break;
           case 2: loop2chunk(lo,hi); break;
         }
      }
    }
  }
}

void loop1chunk(int lo, int hi) {
  int i,j;

  for (i=lo; i<hi; i++){
    for (j=N-1; j>i; j--){
      a[i][j] += cos(b[i][j]);
    }
  }
}



void loop2chunk(int lo, int hi) {
  int i,j,k;
  double rN2;

  rN2 = 1.0 / (double) (N*N);

  for (i=lo; i<hi; i++){
    for (j=0; j < jmax[i]; j++){
      for (k=0; k<j; k++){
	c[i] += (k+1) * log (b[i][j]) * rN2;
      }
    }
  }
}

void valid1(void) {
  int i,j;
  double suma;

  suma= 0.0;
  for (i=0; i<N; i++){
    for (j=0; j<N; j++){
      suma += a[i][j];
    }
  }
  printf("Loop 1 check: Sum of a is %lf\n", suma);
}


void valid2(void) {
  int i;
  double sumc;

  sumc= 0.0;
  for (i=0; i<N; i++){
    sumc += c[i];
  }
  printf("Loop 2 check: Sum of c is %f\n", sumc);
}
