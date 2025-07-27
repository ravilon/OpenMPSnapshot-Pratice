/*
	Copyright 2006 Gabriel Dimitriu

	This file is part of scientific_computing.

    scientific_computing is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    scientific_computing is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with scientific_computing; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  
*/
#include<solve-omp.h>
int gaussJ_fast_omp(long dim,int thread,double **mat,double *x,double *libre)
/*
	dim is dimension of matrix
	thread is the number of threads
	mat is system matrix
	libre is free termen
	x is the solve
*/
{
long s;	// nr of line in partition
long nr;
int dernierre=0;	//if the thread ist the last
long i,j,k,l;
int tid;		//thread identification
int *counters;
long proccount,proccount1,replay;
omp_lock_t *cond_m;
double temp;
	omp_set_num_threads(thread);
	cond_m=(omp_lock_t *)calloc(dim,sizeof(omp_lock_t));
	counters=(int *)calloc(dim,sizeof(int));
	#pragma omp parallel for
	for(i=0;i<dim;i++) 
	{
		omp_init_lock(&cond_m[i]);
		omp_set_lock(&cond_m[i]);
	}
	#pragma omp parallel private(i,j,k,l,s,nr,tid,replay,proccount,proccount1,dernierre,temp)
	{
		tid=omp_get_thread_num();
		dernierre=0;
		nr=dim%thread;
		s=(dim-nr)/thread;
		if((tid+1)<nr) s++;
		else if((tid+1)==nr)
		{
			s++;
			dernierre=1;
		}
		if((nr==0) && ((tid+1)==thread)) dernierre=1;
		//make allocation for the first processor
		if(tid==0)
		{
			#pragma omp atomic
				counters[0]++;
			omp_unset_lock(&cond_m[0]);
			proccount=proccount1=tid+thread;
		}
		else
			proccount=proccount1=tid;
		replay=0;
		// this i is the working line
		for(i=proccount;i<dim;i+=thread)
		{
			//make elimination step
			for(k=replay;k<i;k++)
			{
				if(counters[k]==0)
				{
					omp_set_lock(&cond_m[k]);
					omp_unset_lock(&cond_m[k]);
				}
				temp=mat[i][k]/mat[k][k];
				for(j=(k+1);j<dim;j++)
					mat[i][j]-=mat[k][j]*temp;
				libre[i]-=libre[k]*temp;
				mat[i][k]=0;
			}
			//make settings for next line
			#pragma omp atomic
				counters[i]++;
			omp_unset_lock(&cond_m[i]);
			proccount1=i+thread;
			for(k=replay;k<i;k++)
			{
				for(l=proccount1;l<dim;l+=thread)
				{
					temp=mat[l][k]/mat[k][k];
					for(j=(k+1);j<dim;j++)
						mat[l][j]-=mat[k][j]*temp;
					libre[l]-=libre[k]*temp;
					mat[l][k]=0;
				}
			}
			replay=i;
		}
		
		#pragma omp barrier
		#pragma omp for
		for(i=0;i<dim;i++)
		{
			counters[i]=0;
			omp_set_lock(&cond_m[i]);
		}
		#pragma omp barrier
		if(dernierre==1)
		{
			#pragma omp atomic
				counters[dim-1]++;
			omp_unset_lock(&cond_m[dim-1]);
			proccount=proccount1=dim-thread-1;
		}
		else
			proccount=proccount1=tid+thread*(s-1);
		replay=dim-1;
		// here i is the working line
		for(i=proccount;i>=tid;i-=thread)
		{
			//make elimination step
			for(k=replay;k>i;k--)
			{
				if(counters[k]==0)
				{
					omp_set_lock(&cond_m[k]);
					omp_unset_lock(&cond_m[k]);
				}
				temp=mat[i][k]/mat[k][k];
				for(j=(k+1);j<dim;j++)
					mat[i][j]-=mat[k][j]*temp;
				libre[i]-=libre[k]*temp;
				mat[i][k]=0;
			}
			//make setting for next line
			#pragma omp atomic
				counters[i]++;
			omp_unset_lock(&cond_m[i]);
			proccount1=i-thread;
			for(k=replay;k>i;k--)
			{
				for(l=proccount1;l>=0;l-=thread)
				{
					temp=mat[l][k]/mat[k][k];
					for(j=(k+1);j<dim;j++)
						mat[l][j]-=mat[k][j]*temp;
					libre[l]-=libre[k]*temp;
					mat[l][k]=0;
				}
			}
			replay=i;
		}
	}
	#pragma omp barrier
	#pragma omp parallel for
	for(i=0;i<dim;i++)
		x[i]=libre[i]/mat[i][i];
	#pragma omp parallel for
	for(i=0;i<dim;i++)
		omp_destroy_lock(&cond_m[i]);
	free(cond_m);
	free(counters);
	return 0;
}
