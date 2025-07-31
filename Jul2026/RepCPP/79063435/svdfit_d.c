#define NRANSI


#include "nrutil.h"
#include <stdio.h>
#include <pthread.h>
#define MAX_THREADS 10

#define TOL 1.0e-12

int summer_shared_index;
pthread_mutex_t max_mutex, thresh_mutex, summ_mutex;


struct summ_args { 
	int ma; 
	double *a, *afunc, *sum; 
};


void *dataworker() {

}

void *summer(void * packed_args) {	
	struct summ_args *args = (struct summ_args *)packed_args;
	int ma = args->ma;
	double *a = args->a, *afunc = args->afunc, *sum = args->sum;	
	//printf("1. summer worker args unpacked: ma=%d\n", ma);
	int local_index; 
	double partial_sum = 0;
    	do {
        	pthread_mutex_lock(&summ_mutex);
          	local_index = summer_shared_index;
		summer_shared_index++;
		//printf("2. summer worker: local_index=%d, shared_index=%d, a=%f, afunc=%f\n", 
		//	local_index, summer_shared_index, *(a+local_index), *(afunc+local_index));
        	pthread_mutex_unlock(&summ_mutex);

        	if (local_index <= ma) {
            		partial_sum += (*(a+local_index)) * (*(afunc+local_index));
			//printf("3. partial added local_index=%d: partial=%f = %f * %f\n", 
			//	local_index, partial_sum, *(a+local_index), *(afunc+local_index));
		}
    	} while (local_index <= ma);

	pthread_mutex_lock(&summ_mutex);
	*sum += partial_sum;
	//printf("4. summer worker adding partial=%f to sum=%f\n", partial_sum, *sum);
	pthread_mutex_unlock(&summ_mutex);

	return 0;
}

void svdfit_d(double x[], double y[], double sig[], int ndata, double a[], int ma,
	double **u, double **v, double w[], double *chisq,
	void (*funcs)(double, double [], int))
{
	void svbksb_d(double **u, double w[], double **v, int m, int n, double b[], double x[]);
	void svdcmp_d(double **a, int m, int n, double w[], double **v);
	int j,i,k;
	double wmax,tmp,thresh,sum,*b,*afunc;

	b=dvector(1,ndata);
	afunc=dvector(1,ma);
	
	// WARNGIN: Don't know what afunc is, but needs to be shared().
	#pragma omp parallel for if(ndata > 100)\
		shared(funcs, x, afunc, ma, u, sig, b, y) \
		private(i, j)
	for (i=1;i<=ndata;i++) {	
		(*funcs)(x[i],afunc,ma);
		for (j=1;j<=ma;j++) u[i][j]=afunc[j]/sig[i];
		b[i]=y[i]/sig[i];
	}
	svdcmp_d(u,ndata,ma,w,v);

	// find max val. in w
	wmax=0.0;
	double *wptr = &(w[1]);
	for (j=1;j<=ma;j++, wptr++)
		if (*(wptr) > wmax) wmax=*(wptr);	

	// threshold to zero on all vals. in w < thresh
	thresh=TOL*wmax;
	wptr = &(w[1]);
	for (j=1;j<=ma;j++,wptr++)
		if (*(wptr) < thresh) *(wptr)=0.0;

	svbksb_d(u,w,v,ndata,ma,b,a);

	*chisq=0.0;
	#pragma omp parallel for if(ndata > 100)\
		shared(funcs, x, afunc, ma, a, chisq, y, sig) \
		private(i, j, sum, tmp)
	for (i=1; i<=ndata; i++) {
		(*funcs)(x[i],afunc,ma);
		for (sum=0.0,j=1;j<=ma;j++) sum += a[j] * afunc[j];
		
		// generate MAX_THREADS worker threads to calulate sum
		//pthread_t summ_threads[MAX_THREADS];
		//pthread_mutex_init(&summ_mutex, NULL);
		//sum = 0.0;
		//struct summ_args args;
		//args.ma = ma; 
		//args.a = &(a[0]); args.afunc = &(afunc[0]); args.sum = &sum;
		//summer_shared_index = 1;
		//printf("svdfit_d: generating threads\n");
		//for (k = 0; k < MAX_THREADS; k++) {
    		//	pthread_create(&summ_threads[k] , NULL, summer, (void *)&args);
		//}	
  		//for (k = 0; k < MAX_THREADS; k++) {
    		//	pthread_join(summ_threads[k] , NULL);
		//}
		//printf("svdfit_d: threads collected\n");

		*chisq += (tmp=(y[i]-sum)/sig[i], tmp*tmp);
	}
	free_dvector(afunc,1,ma);
	free_dvector(b,1,ndata);
}


#undef TOL
#undef NRANSI

