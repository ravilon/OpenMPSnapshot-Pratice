/*  Short job 1
*/ 
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define MAX_TEST 5000



int testuj(float *mA,float *mB,float *mC, float *mC2,int ni,int nj,int nk,float a,float b)
{
  int i,j,k,pk,err;
  float pom,k1;
  err=0;
  for(pk=0;pk<MAX_TEST;pk++)
  {    
    i=rand()%ni;
    j=rand()%nj;
    pom=0.0;
    for(k=0;k<nk;k++)                                                                   
    {
      pom+=mA[i*nk+k]*mB[k*nj+j];
    }  
    pom=a*pom+b*mC2[i*nj+j];
    k1=mC[i*nj+j];
    if (fabs(pom-k1)>0.2)  
    {
      err++;
      //printf("%i %i = %g,%g\n",i,j,pom,k1);
    }
  }
  return err;
}
 
  
int vyprazdni(int *temp,int k)
{
  // flush cache
  int i,s=0;
  for(i=0;i<k;i++) s+=temp[i];
  return s;
}  


// zacatek casti k modifikaci
// beginning of part for modification

#define max(X, Y) (((X) > (Y)) ? (X) : (Y))

void compute_my(
	float *inA, float * inB, float * outC, 
	int si, int li, int sj, int lj, int sk, int lk,
	int ni, int nj, int nk, 
	float a, float b) {

	for (int i = si; i < si + li; i++) {
		for (int j = sj; j < sj + lj; j++) {
			float s0 = 0;
			for (int k = sk; k < sk + lk; k++) {
				s0 += inA[i * nk + k] * inB[k * nj + j];
			}
			outC[i * nj + j] = a * s0 + b * outC[i * nj + j];
		}
	}
}

void gemm_my(
	float *inA, float * inB, float * outC, 
	int si, int li, int sj, int lj, int sk, int lk,
	int ni, int nj, int nk, 
	float a, float b) {

	int intA, intB;

	int res = max(li, max(lj, lk));
	long area = ((long)li * lj + li * lk + lj * lk) * 4 * 2;
	if(area <= 64000) {
		// printf("area: %ld\n", area);
		// printf("max: %d\n", res);
		compute_my(inA, inB, outC, si, li, sj, lj, sk, lk, ni, nj, nk, a, b);

	}	else if(res == li) {
		//Split A horizontally - 
		intA = li / 2;
		intB = li - intA;

		// #pragma omp task
		gemm_my(inA, inB, outC, 
			si, intA, sj, lj, sk, lk,
			ni, nj, nk, a, b);
		gemm_my(inA, inB, outC, 
			si + intA, intB, sj, lj, sk, lk,
			ni, nj, nk, a, b);
	} else if(res == lj) {
		//Split B vertically | 
		intA = lj / 2;
		intB = lj - intA;
		// #pragma omp task
		gemm_my(inA, inB, outC, 
			si, li, sj, intA, sk, lk,
			ni, nj, nk, a, b);
		gemm_my(inA, inB, outC, 
			si, li, sj + intA, intB, sk, lk,
			ni, nj, nk, a, b);
	} else if(res == lk) {
		//Split A vertically | 
		//Split B horizontally - 
		intA = lk / 2;
		intB = lk - intA;	
		// #pragma omp task
		gemm_my(inA, inB, outC, 
			si, li, sj, lj, sk, intA,
			ni, nj, nk, a, b);
		gemm_my(inA, inB, outC, 
			si, li, sj, lj, sk + intA, intB,
			ni, nj, nk, a, b);
	} else {
		printf("This should never happened!\n");
	}
}

void gemm( float *inA, float * inB, float * outC, int ni, int nj, int nk, float a, float b ) {
	// #pragma omp parallel
	// #pragma omp single nowait
	gemm_my(inA, inB, outC, 0, ni, 0, nj, 0, nk, ni, nj, nk, a, b);
}
// end of part for modification
// konec casti k modifikaci


int main( void ) {

 double timea[60];
 
 int soucet=0,N,i,j,k,ni,nj,nk,*pomo,v;
 int ri,rj,rk;
 double delta,s_delta=0.0;
 float *mA, *mB,*mC,*mC2; 
    
  // int ti[4]={3500,5000,3200,2500};  
  // int tj[4]={3500,3200,2500,5000};
  // int tk[4]={3500,2500,5000,3200};   
  
  int ti[4]={890,5000,3200,2500};  
  int tj[4]={2050,3200,2500,5000};
  int tk[4]={800,2500,5000,3200}; 

  srand (time(NULL));   
  pomo=(int *)malloc(32*1024*1024);    
  v=0;    
  for(N=0;N<1;N++)
  {
  ni=ti[N];
  nj=tj[N];
  nk=tk[N];

  mA=(float *)malloc(ni * nk* sizeof(float));
  mB=(float *)malloc(nk * nj* sizeof(float));
  mC=(float *)malloc(ni * nj* sizeof(float));
  mC2=(float *)malloc(ni * nj* sizeof(float));

  for (i=0; i<ni; i++) {
  ri=rand();
  for (k=0; k<nk; k++) {
    rk=rand();
    mA[i*nk+k] = (float)(ri%13-6)+(float)(rk%17-8);
  }}
  for (k=0; k<nk; k++) {
    rk=rand();

  for (j=0; j<nj; j++) {
    rj=rand();  
    mB[k*nj+j] = (float)(rk%19-9)+(float)(rj%23-11);
  }}
  for (i=0; i<ni; i++) {
  ri=rand();  
  for (j=0; j<nj; j++) {
    rj=rand();
    mC[i*nj+j] = mC2[i*nj+j] =(float)(ri%29-14)+(float)(rj%31-15);;
  }}


  soucet+=vyprazdni(pomo,v);
  timea[0]=omp_get_wtime();
  // improve performance of this call
  // vylepsit vykonnost tohoto volani
  gemm( mA, mB, mC, ni,nj,nk ,1.0,1.0);
  timea[1]=omp_get_wtime();
  delta=timea[1]-timea[0];
  s_delta+=delta;
  i=testuj(mA,mB,mC,mC2,ni,nj,nk,1.0,1.0);
  printf("%g ",delta);
  j=MAX_TEST;
  if (i!=0) printf("ERR=%i/%i ",i,j);
  fflush(stdout);
 
  free(mC2);
  free(mC);
  free(mB);
  free(mA);
  } 
  printf("%i %g\n",soucet,s_delta); 

  return 0;
}
