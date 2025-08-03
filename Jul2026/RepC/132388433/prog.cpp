//Project 4
//Farm simulator

//Agent I'm adding: Wheat King cat
//TODO add nowCat functions for cat agent

/*cat idea:
every month, there is a random chance for the cat to appear
appearance is random, based on temp and precipitation
if he appears, the deer are shooed away*/
#include "stdio.h"
#include <omp.h>
#include <ctime>
#include <math.h>
#include <stdlib.h>

/*
***Global variables to keep state***
*/
int	NowYear;		// 2017 - 2022
int	NowMonth;		// 0 - 11

double	NowPrecip;		// inches of rain per month
double	NowTemp;		// temperature this month
double	NowHeight;		// grain height in inches
int	    NowNumDeer;		// number of deer in the current population

//TODO add starting cat variable

/*
***Starting variables
Grain height/growth: inches
Precipitation: inches
Temperature: Farenheit 
*/
const double GRAIN_GROWS_PER_MONTH =		8.0;
const double ONE_DEER_EATS_PER_MONTH =		0.5;

const double AVG_PRECIP_PER_MONTH =		6.0;	// average
const double AMP_PRECIP_PER_MONTH =		6.0;	// plus or minus
const double RANDOM_PRECIP =			2.0;	// plus or minus noise

const double AVG_TEMP =				50.0;	// average
const double AMP_TEMP =				20.0;	// plus or minus
const double RANDOM_TEMP =			10.0;	// plus or minus noise

const double MIDTEMP =				40.0;
const double MIDPRECIP =				10.0;
unsigned int seed = 0;  //Seed for RNG

double tempFactor = 0.0;
double precipFactor = 0.0;


/* 
***Helper functions***
*/
//sqrt helper function
double SQR( double x )
{
return x*x;
}

double Ranf( unsigned int *seedp,  double low, double high ){
double r = (double) rand_r( seedp );              // 0 - RAND_MAX
return(   low  +  r * ( high - low ) / (double)RAND_MAX   );
}

int Ranf( unsigned int *seedp, int ilow, int ihigh ){
double low = (double)ilow;
double high = (double)ihigh + 0.9999f;

return (int)(  Ranf(seedp, low,high) );
}

/*Sim functions*/

//Add and fill out each of the 4 sim functions
// while( NowYear < 2023 )
// {
// 	// compute a temporary next-value for this quantity
// 	// based on the current state of the simulation:
// 	. . .

// 	// DoneComputing barrier:
// 	#pragma omp barrier
// 	. . .

// 	// DoneAssigning barrier:
// 	#pragma omp barrier
// 	. . .

// 	// DonePrinting barrier:
// 	#pragma omp barrier
// 	. . .
// }

/*
**temp and precipitation controllers***
*/
void getWeather(){
printf("Hello from Get Weather\n");
// Temp and precipitation are a function of the month
// All months are 30 days, a year is 12 months
// First day of winter = Jan 1
// Temp and precip follow coside and sine waves with some random thrown in
double ang = (  30.*(double)NowMonth + 15.  ) * ( M_PI / 180. );

double temp = AVG_TEMP - AMP_TEMP * cos( ang );
unsigned int seed = 0;
NowTemp = temp + Ranf(&seed, -RANDOM_TEMP, RANDOM_TEMP );

double precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin( ang );
NowPrecip = precip + Ranf( &seed,  -RANDOM_PRECIP, RANDOM_PRECIP );
if( NowPrecip < 0. )
NowPrecip = 0.;

//These determine how close conditions are to "ideal"
tempFactor = exp(   -SQR(  ( NowTemp - MIDTEMP ) / 10.  )   );
precipFactor = exp(   -SQR(  ( NowPrecip - MIDPRECIP ) / 10.  )   );
}

/*Sets the height of the grain each month*/
void Grain(){
printf("Hello from Grain\n");
while( NowYear < 2023 ){
// compute a temporary next-value for grain height
// based on the current state of the simulation:
double curHeight = NowHeight;
curHeight += tempFactor * precipFactor * GRAIN_GROWS_PER_MONTH;
curHeight -= (double)NowNumDeer * ONE_DEER_EATS_PER_MONTH;

if(curHeight < 0){
curHeight = 0;
}
// DoneComputing barrier:
#pragma omp barrier
printf("Done computing grain\n");
//Assign new values to global state variables
NowHeight = curHeight;
//Make sure we don't have negative grain
// DoneAssigning barrier:
#pragma omp barrier

//prinf("Grain height: %f", NowHeight);

// DonePrinting barrier:
#pragma omp barrier
}
}

/*Sets the deer population each month*/
void GrainDeer(){
printf("Hello from Graindeer\n");
while(NowYear < 2023 ){
// compute a temporary next-value for grain height
// based on the current state of the simulation:
double curDeerPop = NowNumDeer;
//For deer, calc carrying capacity based on grain height in previous month
if(curDeerPop > NowHeight){
curDeerPop--;
}
else if (curDeerPop < NowHeight){
curDeerPop ++;
}
else{
curDeerPop == curDeerPop;
}

if(curDeerPop <0){
curDeerPop = 1;
}
// DoneComputing barrier:
#pragma omp barrier
printf("Done computing deer\n");
//Assign new values to global state variables
NowNumDeer = curDeerPop;

// DoneAssigning barrier:
#pragma omp barrier

// DonePrinting barrier:
#pragma omp barrier
}
}

/*Prints current state to console and text file*/
void printState(){
FILE *fp;
fp = fopen("graindeer.txt", "a");

//Time
printf("Month: %d\n",NowMonth);
printf("Year: %d\n",NowYear);

fprintf (fp, "%d\t", NowMonth);
fprintf (fp, "%d\t", NowYear);

//Height and deer
printf("Current height of the grain: %f\n", NowHeight);
printf("Current deer population: %d\n", NowNumDeer);

fprintf (fp, "%f\t", NowHeight);
fprintf (fp, "%d\t", NowNumDeer);

//Temp and precip
printf("Current temp: %f\n", NowTemp);
printf("Current precip: %f\n", NowPrecip);

fprintf (fp, "%f\t", NowTemp);
fprintf (fp, "%f\t", NowPrecip);

fprintf (fp, "\n");
printf("***********\n");
fclose(fp);
}
/*Watches the other threads, prints results, sets weather*/
void Watcher(){
printf("Hello from Watcher\n");
while( NowYear < 2023 ){   
//The watcher function retains 3 barriers
//so that it will be in-step with the
//other threads

// DoneComputing barrier:
#pragma omp barrier

// DoneAssigning barrier:
#pragma omp barrier

//Increment month
NowMonth++;
if(NowMonth >11){
NowYear++;
NowMonth = 0;
}
//Calculate weather
getWeather();

//Print everything
printState();

// DonePrinting barrier:
#pragma omp barrier
}
}

int main( int argc, char *argv[ ] ){

printf("Hello from main!\n");
// Starting date and time:
NowMonth =    0;
NowYear  = 2017;

// starting state:
NowNumDeer = 1;
NowHeight =  1.;

//Set up the weather
getWeather();

//Print initial state
printState();

//Thread zone
omp_set_num_threads(3);	// same as # of sections
#pragma omp parallel sections
{

#pragma omp section
{
printf("About to call GrainDeer()\n");
GrainDeer();
}

#pragma omp section
{
Grain();
}

#pragma omp section
{
Watcher();
}

// 	#pragma omp section
// 	{
// 		//MyAgent( );	// your own
// 	}


}    
printf("We're done!\n");
// implied barrier -- all functions must return in order
// to allow any of them to get past here

return 0;

}