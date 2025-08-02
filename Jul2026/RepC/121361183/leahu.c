#include <stdlib.h>
#include <omp.h>

#define ENABLE_SHOW 1

int Allflags_eq_2(int *f,int n)
{
 int i;
 for (i=0;i<n;i++)
     if (f[i]!=2)
        return 0;
 return 1;
}

void gauss(int n,float **A,float *x,float *b,int thread)
{
   float *y;
   int i,j,k;
   int *flags;
   float *s_l; //shared line -> linia care va avea rol de transmitere de date intre procese
   float s_v_r; //shared value right -> retine valoarea de pe linia de divizare din matricea coloana 'b'
   int s_index; //indexul liniei care a fost facuta broadcast
   int t,divizat=0;
   int tv; //threadul anterior
   int ild;//index linie de divizat
   int did_something; //daca s-a facut o operatie pe linia respectiva

   int  temp;  
   omp_lock_t mtx;
     
   omp_set_num_threads(thread);
   s_l=(float *)calloc(n,sizeof(float));
   flags=(int *)calloc(n,sizeof(int));
   y=(float *)calloc(n,sizeof(float));
    s_index=-1;
    for (i=0;i<n;i++)
    {
       s_l[i]=1;
       flags[i]=0;
       }

  omp_init_lock(&mtx);
  ild=0;
  flags[0]=1;
  do
  {

    #pragma omp parallel for schedule(static,1) shared(y,n,s_l, flags,s_v_r , s_index) private(j,t,did_something,temp)
    for (k=0;k<n;k++)
    {
      t=omp_get_thread_num();
      did_something=0;
      do
      {
         temp=0;
	 omp_set_lock(&mtx);
	 temp=flags[k];
	 omp_unset_lock(&mtx);
         if (temp==1) //divizarea
	  {
	    for (j=k+1;j<n;j++)
	    {
	      A[k][j]=A[k][j]/A[k][k];
	      s_l[j]=A[k][j];
	    }
	    s_l[k]=1;
	    y[k]=b[k]/A[k][k];
	    A[k][k]=1;
	    s_v_r=y[k];
	    s_index=k;
  	    omp_set_lock(&mtx);
	    flags[k]=2;
	    for (j=k+1;j<n;j++)
	       flags[j]=3; //marchez liniile de sub k pentru etapa de eliminare folosind linia 's_l'
	    flags[k+1]=4; //marchez linia k+1 pentru etapa de eliminare , urmata de divizare
	    omp_unset_lock(&mtx);
	    did_something = 1;
	  } //end 'if flags[k]==1'

         if (temp==3) //eliminarea, folosind 'shared_line'
	 {
	   for (j=s_index+1;j<n;j++)
	   {
	     A[k][j]=A[k][j]-A[k][s_index]*s_l[j];

	   }
	   b[k]=b[k]-A[k][s_index]*s_v_r;
	   A[k][s_index]=0;
	   omp_set_lock(&mtx);
	   flags[k]=0; //linia e libera, recalculata
	   omp_unset_lock(&mtx);
           did_something = 1;
	 }
         if (temp==4) //eliminarea, folosind 'shared_line'
	 {
	   for (j=s_index+1;j<n;j++)
	   {
	     A[k][j]=A[k][j]-A[k][s_index]*s_l[j];

	   }
	   b[k]=b[k]-A[k][s_index]*s_v_r;
	   A[k][s_index]=0;
	   omp_set_lock(&mtx);
	   flags[k]=1; //linia e marcata pentru divizare
	   omp_unset_lock(&mtx);
           did_something = 1;
	 }
	 if (temp==2)
	    did_something=1;
        }
       while (!did_something);
      } //end bucla for paralela
      ild++;
   }
   while (!Allflags_eq_2(flags,n)); //toate liniile au fost divizate

  for (k=n-1;k>=0;k--)
   {
	   x[k]=y[k];
	   for(j=k+1;j<n;j++)
	   {
		   x[k]-=A[k][j]*x[j];
	   }
   }
   free(s_l);
   free(y);
   free(flags);
}

