/* 
	file for function of matrix multiplication with chekerboard partition 
	in MPI implementation
*/
#include<stdio.h>
#include<math.h>
#include<parallel/parallel-mpi-omp.h>

int mult_sq_mat_check_mpi_omp(int dim,double **a,double **b,double **c)
/*
	dim c'est la dimension de matrice
	thread c'est le numero de threads
	a c'est la premier matrice
	b c'est la deuxieme matrice
	c c'est la matrice de resoudre
*/
{
int q;	/* c'est la dimension de partition */
int s;	/* c'est le numero de done qui a chacun proces */
int i,j,k,l,m;	/* counteurs */
/* pour le reseau */
MPI_Comm	mesh2_comm;			/* reseau pour faire calcule */
int rank;						/* rank de thread in normal communicator */
int **coordonne;				/* rank de thread in cartesian coordonne */
int size;						/* numero de threads */
int dims[2];					/* les dimensions */
int periodicite[2];			/* la periodicite de reseau */
int *pdone;
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	/* cree le reseau de processors */
	q=sqrt(size);	/* numero de processerus s*s */
	dims[0]=dims[1]=q;
	periodicite[0]=periodicite[1]=0;
	coordonne=(int **)calloc(size,sizeof(int *));
	pdone=(int *)calloc(size*2,sizeof(int));
	for(i=0;i<size;i++)
	{
		coordonne[i]=pdone;
		pdone+=2;
	}
	MPI_Cart_create(MPI_COMM_WORLD,2,dims,periodicite,1,&mesh2_comm);
	MPI_Comm_rank(mesh2_comm,&rank);
	MPI_Cart_coords(mesh2_comm,rank,2,coordonne[0]);
	/* forme le done de transmetre */
	s=dim/q;		/* la dimension de done dans un processeur */
	/* faites les calcules */
	for(l=0;l<s;l++)
	{
		i=s*coordonne[0][0]+l;
		for(m=0;m<s;m++)
		{
			j=s*coordonne[0][1]+m;
			c[i][j]=0.0;
			for(k=0;k<dim;k++)
				c[i][j]+=a[i][k]*b[k][j];
		}
	}
	omp_set_num_threads(size);
#pragma omp parallel private(k,j,l,i)
{
		k=omp_get_thread_num();
		MPI_Cart_coords(mesh2_comm,k,2,coordonne[k]);
		j=s*coordonne[k][1];
		for(l=0;l<s;l++)
		{
			i=s*coordonne[k][0]+l;
			MPI_Bcast(&c[i][j],s,MPI_DOUBLE,k,MPI_COMM_WORLD);
   	}
 }
	/* elibere la memoire */
	free(*coordonne);
	free(coordonne);
	MPI_Comm_free(&mesh2_comm);
	MPI_Barrier(MPI_COMM_WORLD);
}