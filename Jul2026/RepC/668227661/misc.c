/*
 * misc.c
 *
 * Helper functions for
 * - initialization
 * - finalization,
 * - writing out a picture
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "heat.h"

/*
 * Initialize the iterative solver
 * - allocate memory for matrices
 * - set boundary conditions according to configuration
 */
int initialize( algoparam_t *param )
{
    int i, j;
    double dist;

    // total number of points (including border)
    const int np_x = param->act_res_x+2;
	const int np_y = param->act_res_y+2;

    //
    // allocate memory
    //
    (param->u)     = (double*)malloc( sizeof(double)* np_x*np_y );
    (param->uhelp) = (double*)malloc( sizeof(double)* np_x*np_y );
    (param->uvis)  = (double*)calloc( sizeof(double),
				      (param->local_visres_x+2) *
				      (param->local_visres_y+2) );
	// (param->uvis)  = (double*)calloc( sizeof(double),
	// 			      102 * 102);

#pragma omp parallel for schedule(static)
    for (i=0;i<np_y;i++){
    	for (j=0;j<np_x;j++){
    		param->u[i*np_x+j]=0;
			param->uhelp[i*np_x+j]=0;
    	}
    }

    if( !(param->u) || !(param->uhelp) || !(param->uvis) )
    {
	fprintf(stderr, "Error: Cannot allocate memory\n");
	return 0;
    }

    for( i=0; i<param->numsrcs; i++ )
    {	
		/* top row */
		if(param->top_rank < 0)
		{	
			for( j=0; j<np_x; j++ )
			{	
				int tmp = param->x_start + j;
				dist = sqrt( pow((double)tmp/(double)(param->global_res + 1) -
						param->heatsrcs[i].posx, 2)+
					pow(param->heatsrcs[i].posy, 2));

				if( dist <= param->heatsrcs[i].range )
				{

				(param->u)[j] +=
					(param->heatsrcs[i].range-dist) /
					param->heatsrcs[i].range *
					param->heatsrcs[i].temp;
				}
			}
		}
		
		/* bottom row */
		if(param->bottom_rank < 0)
		{
			for( j=0; j<np_x; j++ )
			{
				int tmp = param->x_start + j;
				dist = sqrt( pow((double)tmp/(double)(param->global_res + 1) -
						param->heatsrcs[i].posx, 2)+
					pow(1-param->heatsrcs[i].posy, 2));

				if( dist <= param->heatsrcs[i].range )
				{
				(param->u)[(np_y-1)*np_x+j]+=
					(param->heatsrcs[i].range-dist) /
					param->heatsrcs[i].range *
					param->heatsrcs[i].temp;
				}
			}
		}

		// /* leftmost column */
		if(param->left_rank < 0)
		{
			for( j=1; j<np_y-1; j++ )
			{
				int tmp = param->y_start + j;
				dist = sqrt( pow(param->heatsrcs[i].posx, 2)+
					pow((double)tmp/(double)(param->global_res + 1) -
						param->heatsrcs[i].posy, 2));

				if( dist <= param->heatsrcs[i].range )
				{
				(param->u)[ j*np_x ]+=
					(param->heatsrcs[i].range-dist) /
					param->heatsrcs[i].range *
					param->heatsrcs[i].temp;
				}
			}
		}
		
		// /* rightmost column */
		if(param->right_rank < 0)
		{
			for( j=1; j<np_y-1; j++ )
			{
				int tmp = param->y_start + j;
				dist = sqrt( pow(1-param->heatsrcs[i].posx, 2)+
					pow((double)tmp/(double)(param->global_res + 1) -
						param->heatsrcs[i].posy, 2));

				if( dist <= param->heatsrcs[i].range )
				{
				(param->u)[ j*np_x+(np_x-1) ]+=
					(param->heatsrcs[i].range-dist) /
					param->heatsrcs[i].range *
					param->heatsrcs[i].temp;
				}
			}
		}    
	}

	return 1;
}

/*
 * free used memory
 */
int finalize( algoparam_t *param )
{
    if( param->u ) {
	free(param->u);
	param->u = 0;
    }

    if( param->uhelp ) {
	free(param->uhelp);
	param->uhelp = 0;
    }

    if( param->uvis ) {
	free(param->uvis);
	param->uvis = 0;
    }

    return 1;
}


/*
 * write the given temperature u matrix to rgb values
 * and write the resulting image to file f
 */
void write_image( FILE * f, double *u,
		  unsigned sizex, unsigned sizey )
{
    // RGB table
    unsigned char r[1024], g[1024], b[1024];
    int i, j, k;

    double min, max;

    j=1023;

    // prepare RGB table
    for( i=0; i<256; i++ )
    {
	r[j]=255; g[j]=i; b[j]=0;
	j--;
    }
    for( i=0; i<256; i++ )
    {
	r[j]=255-i; g[j]=255; b[j]=0;
	j--;
    }
    for( i=0; i<256; i++ )
    {
	r[j]=0; g[j]=255; b[j]=i;
	j--;
    }
    for( i=0; i<256; i++ )
    {
	r[j]=0; g[j]=255-i; b[j]=255;
	j--;
    }

    min=DBL_MAX;
    max=-DBL_MAX;

    // find minimum and maximum
    for( i=0; i<sizey; i++ )
    {
	for( j=0; j<sizex; j++ )
	{
	    if( u[i*sizex+j]>max )
		max=u[i*sizex+j];
	    if( u[i*sizex+j]<min )
		min=u[i*sizex+j];
	}
    }

    fprintf(f, "P3\n");
    fprintf(f, "%u %u\n", sizex, sizey);
    fprintf(f, "%u\n", 255);

    for( i=0; i<sizey; i++ )
    {
	for( j=0; j<sizex; j++ )
	{
	    k=(int)(1024.0*(u[i*sizex+j]-min)/(max-min));
	    fprintf(f, "%d %d %d  ", r[k], g[k], b[k]);
	}
	fprintf(f, "\n");
    }
	
}
	

int coarsen( double *uold, unsigned oldx, unsigned oldy ,
	     double *unew, unsigned newx, unsigned newy )
{
    int i, j, k, l, ii, jj;

    int stopx = newx;
    int stopy = newy;
    float temp;
    float stepx = (float) oldx/(float)newx;
    float stepy = (float)oldy/(float)newy;

    if (oldx<newx){
	 stopx=oldx;
	 stepx=1.0;
    }
    if (oldy<newy){
     stopy=oldy;
     stepy=1.0;
    }
    // printf("oldx=%d, newx=%d\n",oldx,newx);
    //printf("oldy=%d, newy=%d\n",oldy,newy);
    //printf("rx=%f, ry=%f\n",stepx,stepy);
    // NOTE: this only takes the top-left corner,
    // and doesnt' do any real coarsening

    for( i=0; i<stopy; i++ ){
       ii=stepy*i;
       for( j=0; j<stopx; j++ ){
          jj=stepx*j;
          temp = 0;
          for ( k=0; k<stepy; k++ ){
	       	for ( l=0; l<stepx; l++ ){
		        temp += uold[(ii+k)*oldx+(jj+l)] ;
	        }
	      }
	      unew[i*newx+j] = temp / (stepy*stepx);
       }
    }

  return 1;
}
