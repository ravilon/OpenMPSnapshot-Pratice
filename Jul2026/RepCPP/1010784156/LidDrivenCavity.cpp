#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cmath>
#include<array>
#include <cstdlib> 
using namespace std;

#include <mpi.h>
#include <omp.h>

#include <cblas.h>

#define IDX(I,J) ((J)*Nx + (I))

//
//sub coords to sub index (Checked)
#define subIDX(i,j) ((j)*subSideNumX + (i))

//

#include "LidDrivenCavity.h"
#include "SolverCG.h"




//convert sub index to global index (CHECKED)
int LidDrivenCavity::subIDXglobal(int iSub, int rank)
{
int iGlobal = 0;
int subRowNum = iSub/ subSideNumXArray[rank];
int coords0 = rank/sideSubCoreNum;
int coords1 = rank%sideSubCoreNum;
iGlobal = coords0 *Nx*subSideNum + subRowNum*Nx + coords1*subSideNum + iSub % subSideNumXArray[rank];
// if (mygrid_rank==3){
//     cout<<"subSideNum at 3 = "<<subSideNum<<endl;
//     cout<<"subSideNumX at 3 = "<<subSideNumX<<endl;
//     cout<<"subSideNumY at 3 = "<<subSideNumY<<endl;
//     cout<<"iSub = "<<iSub<<"; iGlobal = "<<iGlobal<<endl;
// }
return iGlobal;
}



LidDrivenCavity::LidDrivenCavity()
{
}

LidDrivenCavity::~LidDrivenCavity()
{
CleanUp();
}

void LidDrivenCavity::SetDomainSize(double xlen, double ylen)
{
this->Lx = xlen;//Length of the domain in the x-direction
this->Ly = ylen;//Length of the domain in the y-direction
UpdateDxDy();

}

void LidDrivenCavity::SetGridSize(int nx, int ny)
{
this->Nx = nx;//Number of grid points in x-direction
this->Ny = ny;//Number of grid points in y-direction
UpdateDxDy();


}



//Function Divides the grid points for MPI
void LidDrivenCavity::SetSubCore()
{
//    cout<<"CHECKPOINT1"<<endl;



MPI_Comm_rank(MPI_COMM_WORLD, &rank);

MPI_Comm_size(MPI_COMM_WORLD, &size);

//num of process per each side
sideSubCoreNum = sqrt(size);

if (rank == 0 && sideSubCoreNum*sideSubCoreNum != size){
cout<<"ABORTED. NUMBER OF PROCESS MUST BE A SQURE VALUE."<<endl;
MPI_Abort(MPI_COMM_WORLD, 1);
}

//cout << "Process Number = "<< rank+1<< " of " << size << endl;
//Note Example 12.3

//Note Example 12.8
int sizes[2] = {sideSubCoreNum, sideSubCoreNum};
int periods[2] = {0, 0};
int reorder = 1;
MPI_Cart_create(MPI_COMM_WORLD, 2, sizes, periods, reorder, &mygrid);

MPI_Comm_rank(mygrid, &mygrid_rank);
MPI_Cart_coords(mygrid, mygrid_rank, 2, coords);

//num of grid per each side of sub
subSideNum = Nx/sideSubCoreNum;
if (coords[1]==sideSubCoreNum-1){
subSideNumX=Nx/sideSubCoreNum+Nx%sideSubCoreNum;
}
else{
subSideNumX = Nx/sideSubCoreNum;
}
if (coords[0]==sideSubCoreNum-1){
subSideNumY=Ny/sideSubCoreNum+Ny%sideSubCoreNum;
}
else{
subSideNumY = Ny/sideSubCoreNum;
}

//num of grid per sub process
subGridNum = subSideNumX*subSideNumY;
//cout<<"subGridNum = " << subGridNum << endl;

//Record Sub Processes Size at Process 0
if (mygrid_rank!=0){
int subGridNumSend = subGridNum;
MPI_Send(&subGridNumSend, 1, MPI_INT, 0, 100, MPI_COMM_WORLD);
}
else if (mygrid_rank==0){
//store the num of grid of each sub process
subGridNumArray = new int[size]();
subGridNumArray[0] = subGridNum;
for (int i=1; i<size; i++){
int subGridNumRecv=0;
MPI_Recv(&subGridNumRecv, 1, MPI_INT, i, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
subGridNumArray[i]=subGridNumRecv;  
}
}


subSideNumYArray = new int[size]();
subSideNumXArray = new int[size]();
//Record Sub Prcesses Dimensions at Process 0    
if (mygrid_rank!=0){
int subSideNumXSend = subSideNumX;
int subSideNumYSend = subSideNumY;
MPI_Send(&subSideNumXSend, 1, MPI_INT, 0, 101, MPI_COMM_WORLD);
MPI_Send(&subSideNumYSend, 1, MPI_INT, 0, 102, MPI_COMM_WORLD);
}
else if (mygrid_rank==0){
//store the num of grid of each sub process
subSideNumXArray[0] = subSideNumX;
subSideNumYArray[0] = subSideNumY;
for (int i=1; i<size; i++){
int subSideNumXRecv=0;
int subSideNumYRecv=0;
MPI_Recv(&subSideNumXRecv, 1, MPI_INT, i, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(&subSideNumYRecv, 1, MPI_INT, i, 102, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
subSideNumXArray[i]=subSideNumXRecv; 
subSideNumYArray[i]=subSideNumYRecv;  
}
}

MPI_Bcast(subSideNumXArray, size, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(subSideNumYArray, size, MPI_INT, 0, MPI_COMM_WORLD);
// // //Probe: 
// if (mygrid_rank==2){
//     for(int i=0; i<size;i++){
//         cout<<"AT PROCESS "<< i<<", subSideNumXArray = "<<subSideNumXArray[i]<<endl;
//     } 
// }



// // //Probe: 
// if (mygrid_rank==0){
//     for(int i=0; i<size;i++){
//         cout<<"subGridNumArray = "<<subGridNumArray[i]<<endl;
//     } 
// }

// //Probe: Sub Process Dimensions
// if (mygrid_rank==3){
// cout<<"subSideNumX = "<<subSideNumX<<endl;
// cout<<"subSideNumY = "<<subSideNumY<<endl;
// cout<<"subGridNum = " << subGridNum<<endl;
// }


//    cout << "mygrid_rank = " << mygrid_rank << ", coords = " << coords[0] << ", "  << coords[1] <<   "LOCAL I = GLOBAL "<<subIDXglobal(3, mygrid_rank)<<endl;
}
//



void LidDrivenCavity::SetTimeStep(double deltat)
{
this->dt = deltat;
}

void LidDrivenCavity::SetFinalTime(double finalt)
{
this->T = finalt;
}

void LidDrivenCavity::SetReynoldsNumber(double re)
{  
this->Re = re;
this->nu = 1.0/re;
}

void LidDrivenCavity::Initialise()
{
CleanUp();



v   = new double[Npts]();
s   = new double[Npts]();
tmp = new double[Npts]();
cg  = new SolverCG(Nx, Ny, dx, dy);


}




void LidDrivenCavity::Integrate()
{

int NSteps = ceil(T/dt);
for (int t = 0; t < NSteps; ++t)//<NSteps
{
if (mygrid_rank==0){
std::cout << "Step: " << setw(8) << t << "  Time: " << setw(8) << t*dt << std::endl;
//t is Step, t*dt is time
}

Advance(v,s,cg,dx,dy);
}
}





void LidDrivenCavity::WriteSolution(std::string file)
{

// //Probe for s
// if (mygrid_rank==0){
//     cout<< "; s VALUE BEFORE write solution ";
//     for (int i=0; i<Npts; i++){
//         cout<< s[i] << ", ";
//     }
//     cout<<endl;
// }
// //


//Initialize v and s for Sub Cores
double vSub[subGridNum] = {};
double sSub[subGridNum] = {};

#pragma omp parallel for        
//Convert Global v, s to Sub vSub, sSub
for (int i=0;i<subGridNum;i++){
vSub[i] = v[subIDXglobal(i, rank)];
sSub[i] = s[subIDXglobal(i, rank)];
}    


// //Probe for s
// if (mygrid_rank==8){
//     cout<< "; s VALUE BEFORE PRINTING ";
//     for (int i=0; i<Npts; i++){
//         cout<< s[i] << ", ";
//     }
//     cout<<endl;
// }
// //


// //Probe for sSub
// if (mygrid_rank==8){
//     cout<< "; sSub VALUE BEFORE PRINTING ";
//     for (int i=0; i<subGridNum; i++){
//         cout<< sSub[i] << ", ";
//     }
//     cout<<endl;
// }
// //




//Declear neighbour sub exchange buffers
double vSendUp[subSideNumX] = {};
double vSendDown[subSideNumX] = {};
double vSendLeft[subSideNumY] = {};
double vSendRight[subSideNumY] = {};
double sSendUp[subSideNumX] = {};
double sSendDown[subSideNumX] = {};
double sSendLeft[subSideNumY] = {};
double sSendRight[subSideNumY] = {};

double vRecvUp[subSideNumX] = {};
double vRecvDown[subSideNumX] = {};
double vRecvLeft[subSideNumY] = {};
double vRecvRight[subSideNumY] = {};
double sRecvUp[subSideNumX] = {};
double sRecvDown[subSideNumX] = {};
double sRecvLeft[subSideNumY] = {};
double sRecvRight[subSideNumY] = {};



//if sub not at any boundaries
if (coords[0]!=0 && coords[0]!=sideSubCoreNum-1 && coords[1]!=0 && coords[1]!=sideSubCoreNum-1){
for (int i=0; i<subSideNumX; i++){
sSendUp[i] = sSub[i+(subSideNumY-1)*subSideNumX];
sSendDown[i] = sSub[i];
}
for (int i=0; i<subSideNumY; i++){
sSendLeft[i] = sSub[i*subSideNumX];
sSendRight[i] = sSub[subSideNumX-1+i*subSideNumX];
}
MPI_Send(sSendUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 9, MPI_COMM_WORLD);
MPI_Send(sSendDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 10, MPI_COMM_WORLD);
MPI_Send(sSendLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 11, MPI_COMM_WORLD);
MPI_Send(sSendRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 12, MPI_COMM_WORLD);
//cout<<"SENT AT PROCESS: "<<mygrid_rank<<endl;
MPI_Recv(sRecvUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//cout<<"RECEIVD AT PROCESS: "<<mygrid_rank<<endl;
}

//if sub at bottom left corner
else if (coords[0]==0 && coords[1]==0 && sideSubCoreNum > 1){
for (int i=0; i<subSideNumX; i++){
sSendUp[i] = sSub[i+(subSideNumY-1)*subSideNumX];
}
for (int i=0; i<subSideNumY; i++){
sSendRight[i] = sSub[subSideNumX-1+i*subSideNumX];
}
MPI_Send(sSendUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 9, MPI_COMM_WORLD);
MPI_Send(sSendRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 12, MPI_COMM_WORLD);
//cout<<"SENT AT PROCESS: "<<mygrid_rank<<endl;
MPI_Recv(sRecvUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//cout<<"RECEIVD AT PROCESS: "<<mygrid_rank<<endl;
}

//if sub at bottom right corner
else if (coords[0]==0 && coords[1]==sideSubCoreNum-1 && sideSubCoreNum>1){
for (int i=0; i<subSideNumX; i++){
sSendUp[i] = sSub[i+(subSideNumY-1)*subSideNumX];
}
for (int i=0; i<subSideNumY; i++){
sSendLeft[i] = sSub[i*subSideNumX];
}
MPI_Send(sSendUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 9, MPI_COMM_WORLD);
MPI_Send(sSendLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 11, MPI_COMM_WORLD);
//cout<<"SENT AT PROCESS: "<<mygrid_rank<<endl;
MPI_Recv(sRecvUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//cout<<"RECEIVD AT PROCESS: "<<mygrid_rank<<endl;
}

//if sub at top left corner
else if (coords[0]==sideSubCoreNum-1 && coords[1]==0 && sideSubCoreNum>1){
for (int i=0; i<subSideNumX; i++){
sSendDown[i] = sSub[i];
}
for (int i=0; i<subSideNumY; i++){
sSendRight[i] = sSub[subSideNumX-1+i*subSideNumX];
}
MPI_Send(sSendDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 10, MPI_COMM_WORLD);
MPI_Send(sSendRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 12, MPI_COMM_WORLD);
//cout<<"SENT AT PROCESS: "<<mygrid_rank<<endl;
MPI_Recv(sRecvDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//cout<<"RECEIVD AT PROCESS: "<<mygrid_rank<<endl;
}

//if sub at top right corner
else if (coords[0]==sideSubCoreNum-1 && coords[1]==sideSubCoreNum-1 && sideSubCoreNum>1){
for (int i=0; i<subSideNumX; i++){
sSendDown[i] = sSub[i];
}
for (int i=0; i<subSideNumY; i++){
sSendLeft[i] = sSub[i*subSideNumX];
}
MPI_Send(sSendDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 10, MPI_COMM_WORLD);
MPI_Send(sSendLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 11, MPI_COMM_WORLD);
//cout<<"SENT AT PROCESS: "<<mygrid_rank<<endl;
MPI_Recv(sRecvDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//cout<<"RECEIVD AT PROCESS: "<<mygrid_rank<<endl;
}

//if sub at top edge
else if (coords[0]==sideSubCoreNum-1 && coords[1]!=sideSubCoreNum-1 && coords[1]!=0){
for (int i=0; i<subSideNumX; i++){
sSendDown[i] = sSub[i];
}
for (int i=0; i<subSideNumY; i++){
sSendLeft[i] = sSub[i*subSideNumX];
sSendRight[i] = sSub[subSideNumX-1+i*subSideNumX];
}
MPI_Send(sSendDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 10, MPI_COMM_WORLD);
MPI_Send(sSendLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 11, MPI_COMM_WORLD);
MPI_Send(sSendRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 12, MPI_COMM_WORLD);
//cout<<"SENT AT PROCESS: "<<mygrid_rank<<endl;
MPI_Recv(sRecvDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//cout<<"RECEIVD AT PROCESS: "<<mygrid_rank<<endl;
}

//if sub at bottom edge
else if (coords[0]==0 && coords[1]!=sideSubCoreNum-1 && coords[1]!=0){
for (int i=0; i<subSideNumX; i++){
sSendUp[i] = sSub[i+(subSideNumY-1)*subSideNumX];
}
for (int i=0; i<subSideNumY; i++){
sSendLeft[i] = sSub[i*subSideNumX];
sSendRight[i] = sSub[subSideNumX-1+i*subSideNumX];
}
MPI_Send(sSendUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 9, MPI_COMM_WORLD);
MPI_Send(sSendLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 11, MPI_COMM_WORLD);
MPI_Send(sSendRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 12, MPI_COMM_WORLD);
//cout<<"SENT AT PROCESS: "<<mygrid_rank<<endl;
MPI_Recv(sRecvUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//cout<<"RECEIVD AT PROCESS: "<<mygrid_rank<<endl;
}

//if sub at right edge
else if (coords[1]==sideSubCoreNum-1 && coords[0]!=sideSubCoreNum-1 && coords[0]!=0){
for (int i=0; i<subSideNumX; i++){
sSendUp[i] = sSub[i+(subSideNumY-1)*subSideNumX];
sSendDown[i] = sSub[i];
}
for (int i=0; i<subSideNumY; i++){
sSendLeft[i] = sSub[i*subSideNumX];
}
MPI_Send(sSendUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 9, MPI_COMM_WORLD);
MPI_Send(sSendDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 10, MPI_COMM_WORLD);
MPI_Send(sSendLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 11, MPI_COMM_WORLD);
//cout<<"SENT AT PROCESS: "<<mygrid_rank<<endl;
MPI_Recv(sRecvUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//cout<<"RECEIVD AT PROCESS: "<<mygrid_rank<<endl;
}
//if sub at left edge
else if (coords[1]==0 && coords[0]!=sideSubCoreNum-1 && coords[0]!=0){
for (int i=0; i<subSideNumX; i++){
sSendUp[i] = sSub[i+(subSideNumY-1)*subSideNumX];
sSendDown[i] = sSub[i];
}
for (int i=0; i<subSideNumY; i++){
sSendRight[i] = sSub[subSideNumX-1+i*subSideNumX];
}
MPI_Send(sSendUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 9, MPI_COMM_WORLD);
MPI_Send(sSendDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 10, MPI_COMM_WORLD);
MPI_Send(sSendRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 12, MPI_COMM_WORLD);
//cout<<"SENT AT PROCESS: "<<mygrid_rank<<endl;
MPI_Recv(sRecvUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvRight,subSideNumY, MPI_DOUBLE, mygrid_rank+1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//cout<<"RECEIVD AT PROCESS: "<<mygrid_rank<<endl;
}





//test sub position when alogrithm nelegecting boundary
int iBegin = 99999;
int iEnd = 99999;
int jBegin = 99999;
int jEnd = 99999; 

//if sub not at any boundaries
if (coords[0]!=0 && coords[0]!=sideSubCoreNum-1 && coords[1]!=0 && coords[1]!=sideSubCoreNum-1){
iBegin = 0;
iEnd = subSideNumX;
jBegin = 0;
jEnd = subSideNumY; 
}
//if sub at bottom left corner
else if (coords[0]==0 && coords[1]==0){
iBegin = 1;
iEnd = subSideNumX;
jBegin = 1;
jEnd = subSideNumY; 
}
//if sub at bottom right corner
else if (coords[0]==0 && coords[1]==sideSubCoreNum-1){
iBegin = 0;
iEnd = subSideNumX-1;
jBegin = 1;
jEnd = subSideNumY; 
}
//if sub at top left corner
else if (coords[0]==sideSubCoreNum-1 && coords[1]==0){
iBegin = 1;
iEnd = subSideNumX;
jBegin = 0;
jEnd = subSideNumY-1; 
}
//if sub at top right corner
else if (coords[0]==sideSubCoreNum-1 && coords[1]==sideSubCoreNum-1){
iBegin = 0;
iEnd = subSideNumX-1;
jBegin = 0;
jEnd = subSideNumY-1; 
}
//if sub at top edge
else if (coords[0]==sideSubCoreNum-1 && coords[1]!=sideSubCoreNum-1 && coords[1]!=0){
iBegin = 0;
iEnd = subSideNumX;
jBegin = 0;
jEnd = subSideNumY-1; 
}
//if sub at bottom edge
else if (coords[0]==0 && coords[1]!=sideSubCoreNum-1 && coords[1]!=0){
iBegin = 0;
iEnd = subSideNumX;
jBegin = 1;
jEnd = subSideNumY; 
}
//if sub at right edge
else if (coords[1]==sideSubCoreNum-1 && coords[0]!=sideSubCoreNum-1 && coords[0]!=0){
iBegin = 0;
iEnd = subSideNumX-1;
jBegin = 0;
jEnd = subSideNumY; 
}
//if sub at left edge
else if (coords[1]==0 && coords[0]!=sideSubCoreNum-1 && coords[0]!=0){
iBegin = 1;
iEnd = subSideNumX;
jBegin = 0;
jEnd = subSideNumY; 
}
else if (size == 1){
iBegin = 1;
iEnd = subSideNumX-1;
jBegin = 1;
jEnd = subSideNumY-1;    
}


//initialize u0Sub u1Sub
double u0Sub[subGridNum] = {};
double u1Sub[subGridNum] = {};

double siPlus = 0;
double siMinus = 0;
double sjPlus = 0;
double sjMinus = 0;
double viPlus = 0;
double viMinus = 0;
double vjPlus = 0;
double vjMinus = 0;

int nthreads, threadid;
#pragma omp parallel for default(shared) private(siPlus,siMinus,sjPlus,sjMinus,viPlus,viMinus,vjPlus,vjMinus) //calculate u0Sub, u1Sub from stream function sSub 
for (int i = iBegin; i < iEnd; ++i) {
for (int j = jBegin; j < jEnd; ++j) {

siPlus = sSub[subIDX(i+1,j)];
siMinus = sSub[subIDX(i-1,j)];
sjPlus = sSub[subIDX(i,j+1)];
sjMinus = sSub[subIDX(i,j-1)];

//Test if need to receive data from neigbour sub
if (i+1>subSideNumX-1){
siPlus = sRecvRight[j];
}
if (i-1<0){
siMinus = sRecvLeft[j];
}
if (j+1>subSideNumY-1){
sjPlus = sRecvUp[i];
}
if (j-1<0){
sjMinus = sRecvDown[i];
}


u0Sub[subIDX(i,j)] =  (sjPlus - sSub[subIDX(i,j)]) / dy;
u1Sub[subIDX(i,j)] = -(siPlus - sSub[subIDX(i,j)]) / dx;
}
}



//SET TOP BOUNDARY X-VELOCITY
if (coords[0] == sideSubCoreNum-1){
for (int i = 0; i < subSideNumX; ++i) {
u0Sub[subIDX(i,subSideNumY-1)] = U;
//cout<<"coords = " << coords[0] << ", "  << coords[1] <<"u0Sub[subIDX(i,subSideNum-1)] = "<< u0Sub[subIDX(i,subSideNum-1)]<<endl;
}
}



// //Probe for s
// if (mygrid_rank==4){
//     cout<< "; u0Sub VALUE BEFORE PRINTING ";
//     for (int i=0; i<subGridNum; i++){
//         cout<< u0Sub[i] << ", ";
//     }
//     cout<<endl;
// }
// //



//Collect u0Sub, u1Sub to u0, u1 Global
//If not at Process 0, send u0Sub, u1Sub to Process 0
if (mygrid_rank != 0){

MPI_Send(u0Sub, subGridNum, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
MPI_Send(u1Sub, subGridNum, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
//cout<<"SENT FROM "<< mygrid_rank << " VALUE = " << u0Sub[6]<<endl;
}

//If at process 0
if (mygrid_rank == 0){
//cout<<"PROCESS 0 BEGINS"<<endl;

double* u0 = new double[Nx*Ny]();
double* u1 = new double[Nx*Ny]();       

//for All Process
for (int i = 0; i < size; i++) {

//Process 0
if(i==0){
#pragma omp parallel for        
for (int j=0; j<subGridNum; j++){
u0[subIDXglobal(j,0)] = u0Sub[j];
u1[subIDXglobal(j,0)] = u1Sub[j];
}
}
//Other Processes
else if (i != 0){
double u0temp[subGridNumArray[i]] = {};
double u1temp[subGridNumArray[i]] = {};
//cout<<"u0temp"<<u0temp[4]<<endl;
MPI_Recv(u0temp, subGridNumArray[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(u1temp, subGridNumArray[i], MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

//cout<<"RECEIVED FROM "<< i << " VALUE = " << u0temp[7]<<endl;

#pragma omp parallel for        
for (int j=0; j<subGridNumArray[i]; j++){
u0[subIDXglobal(j,i)] = u0temp[j];
u1[subIDXglobal(j,i)] = u1temp[j];
}                
}
}
//Collect u0Sub, u1Sub to u0, u1 Global



// //Print u0
// for (int j=0; j<Nx*Ny; j++){
//     cout<<u0[j];
// }

std::ofstream f(file.c_str());
std::cout << "Writing file " << file << std::endl;
int k = 0;

for (int i = 0; i < Nx; ++i)
{
for (int j = 0; j < Ny; ++j)
{
k = IDX(i, j);
f << i * dx << " " << j * dy << " " << v[k] <<  " " << s[k] 
<< " " << u0[k] << " " << u1[k] << std::endl;
}
f << std::endl;
}
f.close();

delete[] u0;
delete[] u1;
}

}





void LidDrivenCavity::PrintConfiguration()
{
if (mygrid_rank == 0){
cout << "Grid size: " << Nx << " x " << Ny << endl;
cout << "Spacing:   " << dx << " x " << dy << endl;
cout << "Length:    " << Lx << " x " << Ly << endl;
cout << "Grid pts:  " << Npts << endl;
cout << "Timestep:  " << dt << endl;
cout << "Steps:     " << ceil(T/dt) << endl;
cout << "Reynolds number: " << Re << endl;
cout << "Linear solver: preconditioned conjugate gradient" << endl;
cout << endl;
}
if (nu * dt / dx / dy > 0.25) {
cout << "ERROR: Time-step restriction not satisfied!" << endl;
cout << "Maximum time-step is " << 0.25 * dx * dy / nu << endl;
exit(-1);
}
}


void LidDrivenCavity::CleanUp()
{
if (v) {
delete[] v;
delete[] s;
delete[] tmp;
delete cg;
}
}




//Npts, dx, dy
void LidDrivenCavity::UpdateDxDy()
{
dx = Lx / (Nx-1);
dy = Ly / (Ny-1);
Npts = Nx * Ny;
}





void LidDrivenCavity::Advance(double* v,double* s, SolverCG* cg, double dx, double dy)
{
double dxi  = 1.0/dx;
double dyi  = 1.0/dy;
double dx2i = 1.0/dx/dx;
double dy2i = 1.0/dy/dy;

// //创建81个增值排列的array v用于测试
// for (int i = 0; i < 81; ++i) {
//     v[i] = i;
// }


//Initialize v and s for Sub Cores
double vSub[subGridNum] = {};
double sSub[subGridNum] = {};

//  int nthreads, threadid;
#pragma omp parallel for 
//Convert Global v, s to Sub vSub, sSub
for (int i=0;i<subGridNum;i++){
vSub[i] = v[subIDXglobal(i, rank)];
sSub[i] = s[subIDXglobal(i, rank)];
// threadid =  omp_get_thread_num();
// cout<<"threadid = "<<threadid<<endl;
// nthreads = omp_get_num_threads();
// cout<<"nthreads = "<<nthreads<<endl;
}    

// //Print vSub
// if (mygrid_rank==0){
//     cout<<"THIS IS PROCESS "<< 8<< endl;
//         for (int j=0; j<subGridNum; j++){
//             cout<<vSub[j]<<", ";
//         }
// }
//cout<<"dy2i = "<<dy2i<<endl;


//--------------------------------------- Boundary node vorticity-----------------------------------
if (size == 1){
// Boundary node vorticity
for (int i = 1; i < Nx-1; ++i) {
// top
vSub[IDX(i,0)]    = 2.0 * dy2i * (sSub[IDX(i,0)]    - sSub[IDX(i,1)]);
// bottom
vSub[IDX(i,Ny-1)] = 2.0 * dy2i * (sSub[IDX(i,Ny-1)] - sSub[IDX(i,Ny-2)])
- 2.0 * dyi*U;
}
}
//if core is at top or bottom edges, not at left or right corner
else if (coords[1] != 0 && coords[1] != sideSubCoreNum-1 ){
for (int i = 0; i < subSideNumX; ++i) {
// bottom
if (coords[0]==0){
vSub[subIDX(i,0)]    = 2.0 * dy2i * (sSub[subIDX(i,0)]    - sSub[subIDX(i,1)]);
}

// top
if (coords[0]==sideSubCoreNum-1){
vSub[subIDX(i,subSideNumY-1)] = 2.0 * dy2i * (sSub[subIDX(i,subSideNumY-1)] - sSub[subIDX(i,subSideNumY-2)]) - 2.0 * dyi*U;
}
}
}
//if core is at left corners
else if (coords[1] == 0){
for (int i = 1; i < subSideNumX; ++i) {
// bottom
if (coords[0]==0){
vSub[subIDX(i,0)]    = 2.0 * dy2i * (sSub[subIDX(i,0)]    - sSub[subIDX(i,1)]);
}

// top
if (coords[0]==sideSubCoreNum-1){
vSub[subIDX(i,subSideNumY-1)] = 2.0 * dy2i * (sSub[subIDX(i,subSideNumY-1)] - sSub[subIDX(i,subSideNumY-2)]) - 2.0 * dyi*U;
}
}
}
//if core is at right corners
else if (coords[1] == sideSubCoreNum-1){
for (int i = 0; i < subSideNumX-1; ++i) {
// bottom
if (coords[0]==0){
vSub[subIDX(i,0)]    = 2.0 * dy2i * (sSub[subIDX(i,0)]    - sSub[subIDX(i,1)]);
}

// top
if (coords[0]==sideSubCoreNum-1){
vSub[subIDX(i,subSideNumY-1)] = 2.0 * dy2i * (sSub[subIDX(i,subSideNumY-1)] - sSub[subIDX(i,subSideNumY-2)]) - 2.0 * dyi*U;
}
}
}



//if core is at left or right edges, not at bottom or top corners
if (coords[0] != 0 && coords[0] != sideSubCoreNum-1 ){ 
for (int j = 0; j < subSideNumY; ++j) {
// left
if (coords[1]==0){
vSub[subIDX(0,j)] = 2.0 * dx2i * (sSub[subIDX(0,j)]    - sSub[subIDX(1,j)]);
}

// right
if (coords[1]==sideSubCoreNum-1){
vSub[subIDX(subSideNumX-1,j)] = 2.0 * dx2i * (sSub[subIDX(subSideNumX-1,j)] - sSub[subIDX(subSideNumX-2,j)]);
}
}
}
//if core is at bottom corners
else if (coords[0] == 0){ 
for (int j = 1; j < subSideNumY; ++j) {
// left
if (coords[1]==0){
vSub[subIDX(0,j)] = 2.0 * dx2i * (sSub[subIDX(0,j)]    - sSub[subIDX(1,j)]);
}

// right
if (coords[1]==sideSubCoreNum-1){
vSub[subIDX(subSideNumX-1,j)] = 2.0 * dx2i * (sSub[subIDX(subSideNumX-1,j)] - sSub[subIDX(subSideNumX-2,j)]);
}
}
}
//if core is at top corners
else if (coords[0] == sideSubCoreNum-1){ 
for (int j = 1; j < subSideNumY; ++j) {
// left
if (coords[1]==0){
vSub[subIDX(0,j)] = 2.0 * dx2i * (sSub[subIDX(0,j)]    - sSub[subIDX(1,j)]);
}

// right
if (coords[1]==sideSubCoreNum-1){
vSub[subIDX(subSideNumX-1,j)] = 2.0 * dx2i * (sSub[subIDX(subSideNumX-1,j)] - sSub[subIDX(subSideNumX-2,j)]);
}
}
}
//--------------------------------------- Boundary node vorticity END--------------------------------------- 



//PROBE
// //Gather Sub v to Global at Processor 0 
//     //If not at Process 0, send u0Sub, u1Sub to Process 0
//     if (mygrid_rank != 0){
//         MPI_Send(vSub, subGridNum, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
//         //cout<<"s SENT FROM "<< mygrid_rank << " VALUE = " << sSub[7]<<endl;
//     }

//     //If at process 0
//     if (mygrid_rank == 0){
//         //cout<<"PROCESS 0 FOR V AND S BEGINS"<<endl;

//         //for All Process
//         for (int i = 0; i < size; i++) {

//             //Process 0
//             if(i==0){
//                 for (int j=0; j<subGridNum; j++){
//                    v[subIDXglobal(j,0)] = vSub[j];
//                 }
//             }

//             //At Other Processes
//             else if (i != 0){
//                 double vtemp[subGridNumArray[i]] = {};

//                 MPI_Recv(vtemp, subGridNumArray[i], MPI_DOUBLE, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//                 //cout<<"s RECEIVED FROM "<< i << " VALUE = " << stemp[7]<<endl;

//                 for (int j=0; j<subGridNumArray[i]; j++){
//                     v[subIDXglobal(j,i)] = vtemp[j];
//                 }                
//             }
//         }
//     }
//     //Gather Sub v to Global at Processor 0




//     //Probe for s
//     if (mygrid_rank==0){
//         cout<< "; s VALUE BEFORE INTERIOR= ";
//         for (int i=0; i<Npts; i++){
//             cout<< s[i] << ", ";
//         }
//         cout<<endl;
//     }
//     //

//     //Probe for v
//     if (mygrid_rank==0){
//         cout<< "; v VALUE BEFORE INTERIOR= ";
//         for (int i=0; i<Npts; i++){
//             cout<< v[i] << ", ";
//         }
//         cout<<endl;
//     }
//     //





//test sub position when alogrithm nelegecting boundary
int iBegin = 99999;
int iEnd = 99999;
int jBegin = 99999;
int jEnd = 99999; 

//if sub not at any boundaries
if (coords[0]!=0 && coords[0]!=sideSubCoreNum-1 && coords[1]!=0 && coords[1]!=sideSubCoreNum-1){
iBegin = 0;
iEnd = subSideNumX;
jBegin = 0;
jEnd = subSideNumY; 
}
//if sub at bottom left cornerdy
else if (coords[0]==0 && coords[1]==0 && size>1){
iBegin = 1;
iEnd = subSideNumX;
jBegin = 1;
jEnd = subSideNumY; 
}
//if sub at bottom right corner
else if (coords[0]==0 && coords[1]==sideSubCoreNum-1 && size>1){
iBegin = 0;
iEnd = subSideNumX-1;
jBegin = 1;
jEnd = subSideNumY; 
}
//if sub at top left corner
else if (coords[0]==sideSubCoreNum-1 && coords[1]==0 && size>1){
iBegin = 1;
iEnd = subSideNumX;
jBegin = 0;
jEnd = subSideNumY-1; 
}
//if sub at top right corner
else if (coords[0]==sideSubCoreNum-1 && coords[1]==sideSubCoreNum-1 && size>1){
iBegin = 0;
iEnd = subSideNumX-1;
jBegin = 0;
jEnd = subSideNumY-1; 
}
//if sub at top edge
else if (coords[0]==sideSubCoreNum-1 && coords[1]!=sideSubCoreNum-1 && coords[1]!=0){
iBegin = 0;
iEnd = subSideNumX;
jBegin = 0;
jEnd = subSideNumY-1; 
}
//if sub at bottom edge
else if (coords[0]==0 && coords[1]!=sideSubCoreNum-1 && coords[1]!=0){
iBegin = 0;
iEnd = subSideNumX;
jBegin = 1;
jEnd = subSideNumY; 
}
//if sub at right edge
else if (coords[1]==sideSubCoreNum-1 && coords[0]!=sideSubCoreNum-1 && coords[0]!=0){
iBegin = 0;
iEnd = subSideNumX-1;
jBegin = 0;
jEnd = subSideNumY; 
}
//if sub at left edge
else if (coords[1]==0 && coords[0]!=sideSubCoreNum-1 && coords[0]!=0){
iBegin = 1;
iEnd = subSideNumX;
jBegin = 0;
jEnd = subSideNumY; 
}
else if (size == 1){
iBegin = 1;
iEnd = subSideNumX-1;
jBegin = 1;
jEnd = subSideNumY-1;    
}

// //Test Probe for Above Code
// cout << "mygrid_rank = " << mygrid_rank << ", coords = " << coords[0] << ", "  << coords[1] << ", iBegin = " << iBegin 
// << ", iEnd = " << iEnd << ", jBegin = " << jBegin << ", jEnd = " << jEnd <<endl;










//Declear neighbour sub exchange buffers
double vSendUp[subSideNumX] = {};
double vSendDown[subSideNumX] = {};
double vSendLeft[subSideNumY] = {};
double vSendRight[subSideNumY] = {};
double sSendUp[subSideNumX] = {};
double sSendDown[subSideNumX] = {};
double sSendLeft[subSideNumY] = {};
double sSendRight[subSideNumY] = {};

double vRecvUp[subSideNumX] = {};
double vRecvDown[subSideNumX] = {};
double vRecvLeft[subSideNumY] = {};
double vRecvRight[subSideNumY] = {};
double sRecvUp[subSideNumX] = {};
double sRecvDown[subSideNumX] = {};
double sRecvLeft[subSideNumY] = {};
double sRecvRight[subSideNumY] = {};


//if sub not at any boundaries
if (coords[0]!=0 && coords[0]!=sideSubCoreNum-1 && coords[1]!=0 && coords[1]!=sideSubCoreNum-1){
for (int i=0; i<subSideNumX; i++){
sSendUp[i] = sSub[i+(subSideNumY-1)*subSideNumX];
sSendDown[i] = sSub[i];
}
for (int i=0; i<subSideNumY; i++){
sSendLeft[i] = sSub[i*subSideNumX];
sSendRight[i] = sSub[subSideNumX-1+i*subSideNumX];
}
MPI_Send(sSendUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 9, MPI_COMM_WORLD);
MPI_Send(sSendDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 10, MPI_COMM_WORLD);
MPI_Send(sSendLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 11, MPI_COMM_WORLD);
MPI_Send(sSendRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 12, MPI_COMM_WORLD);
//cout<<"SENT AT PROCESS: "<<mygrid_rank<<endl;
MPI_Recv(sRecvUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//cout<<"RECEIVD AT PROCESS: "<<mygrid_rank<<endl;
}

//if sub at bottom left corner
else if (coords[0]==0 && coords[1]==0 && sideSubCoreNum>1){
for (int i=0; i<subSideNumX; i++){
sSendUp[i] = sSub[i+(subSideNumY-1)*subSideNumX];
}
for (int i=0; i<subSideNumY; i++){
sSendRight[i] = sSub[subSideNumX-1+i*subSideNumX];
}
MPI_Send(sSendUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 9, MPI_COMM_WORLD);
MPI_Send(sSendRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 12, MPI_COMM_WORLD);
//cout<<"SENT AT PROCESS: "<<mygrid_rank<<endl;
MPI_Recv(sRecvUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//cout<<"RECEIVD AT PROCESS: "<<mygrid_rank<<endl;
}

//if sub at bottom right corner
else if (coords[0]==0 && coords[1]==sideSubCoreNum-1 && sideSubCoreNum>1){
for (int i=0; i<subSideNumX; i++){
sSendUp[i] = sSub[i+(subSideNumY-1)*subSideNumX];
}
for (int i=0; i<subSideNumY; i++){
sSendLeft[i] = sSub[i*subSideNumX];
}
MPI_Send(sSendUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 9, MPI_COMM_WORLD);
MPI_Send(sSendLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 11, MPI_COMM_WORLD);
//cout<<"SENT AT PROCESS: "<<mygrid_rank<<endl;
MPI_Recv(sRecvUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//cout<<"RECEIVD AT PROCESS: "<<mygrid_rank<<endl;
}

//if sub at top left corner
else if (coords[0]==sideSubCoreNum-1 && coords[1]==0 && sideSubCoreNum>1){
for (int i=0; i<subSideNumX; i++){
sSendDown[i] = sSub[i];
}
for (int i=0; i<subSideNumY; i++){
sSendRight[i] = sSub[subSideNumX-1+i*subSideNumX];
}
MPI_Send(sSendDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 10, MPI_COMM_WORLD);
MPI_Send(sSendRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 12, MPI_COMM_WORLD);
//cout<<"SENT AT PROCESS: "<<mygrid_rank<<endl;
MPI_Recv(sRecvDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//cout<<"RECEIVD AT PROCESS: "<<mygrid_rank<<endl;
}

//if sub at top right corner
else if (coords[0]==sideSubCoreNum-1 && coords[1]==sideSubCoreNum-1 && sideSubCoreNum>1){
for (int i=0; i<subSideNumX; i++){
sSendDown[i] = sSub[i];
}
for (int i=0; i<subSideNumY; i++){
sSendLeft[i] = sSub[i*subSideNumX];
}
MPI_Send(sSendDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 10, MPI_COMM_WORLD);
MPI_Send(sSendLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 11, MPI_COMM_WORLD);
//cout<<"SENT AT PROCESS: "<<mygrid_rank<<endl;
MPI_Recv(sRecvDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//cout<<"RECEIVD AT PROCESS: "<<mygrid_rank<<endl;
}

//if sub at top edge
else if (coords[0]==sideSubCoreNum-1 && coords[1]!=sideSubCoreNum-1 && coords[1]!=0){
for (int i=0; i<subSideNumX; i++){
sSendDown[i] = sSub[i];
}
for (int i=0; i<subSideNumY; i++){
sSendLeft[i] = sSub[i*subSideNumX];
sSendRight[i] = sSub[subSideNumX-1+i*subSideNumX];
}
MPI_Send(sSendDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 10, MPI_COMM_WORLD);
MPI_Send(sSendLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 11, MPI_COMM_WORLD);
MPI_Send(sSendRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 12, MPI_COMM_WORLD);
//cout<<"SENT AT PROCESS: "<<mygrid_rank<<endl;
MPI_Recv(sRecvDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//cout<<"RECEIVD AT PROCESS: "<<mygrid_rank<<endl;
}

//if sub at bottom edge
else if (coords[0]==0 && coords[1]!=sideSubCoreNum-1 && coords[1]!=0){
for (int i=0; i<subSideNumX; i++){
sSendUp[i] = sSub[i+(subSideNumY-1)*subSideNumX];
}
for (int i=0; i<subSideNumY; i++){
sSendLeft[i] = sSub[i*subSideNumX];
sSendRight[i] = sSub[subSideNumX-1+i*subSideNumX];
}
MPI_Send(sSendUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 9, MPI_COMM_WORLD);
MPI_Send(sSendLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 11, MPI_COMM_WORLD);
MPI_Send(sSendRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 12, MPI_COMM_WORLD);
//cout<<"SENT AT PROCESS: "<<mygrid_rank<<endl;
MPI_Recv(sRecvUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//cout<<"RECEIVD AT PROCESS: "<<mygrid_rank<<endl;
}

//if sub at right edge
else if (coords[1]==sideSubCoreNum-1 && coords[0]!=sideSubCoreNum-1 && coords[0]!=0){
for (int i=0; i<subSideNumX; i++){
sSendUp[i] = sSub[i+(subSideNumY-1)*subSideNumX];
sSendDown[i] = sSub[i];
}
for (int i=0; i<subSideNumY; i++){
sSendLeft[i] = sSub[i*subSideNumX];
}
MPI_Send(sSendUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 9, MPI_COMM_WORLD);
MPI_Send(sSendDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 10, MPI_COMM_WORLD);
MPI_Send(sSendLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 11, MPI_COMM_WORLD);
//cout<<"SENT AT PROCESS: "<<mygrid_rank<<endl;
MPI_Recv(sRecvUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//cout<<"RECEIVD AT PROCESS: "<<mygrid_rank<<endl;
}
//if sub at left edge
else if (coords[1]==0 && coords[0]!=sideSubCoreNum-1 && coords[0]!=0){
for (int i=0; i<subSideNumX; i++){
sSendUp[i] = sSub[i+(subSideNumY-1)*subSideNumX];
sSendDown[i] = sSub[i];
}
for (int i=0; i<subSideNumY; i++){
sSendRight[i] = sSub[subSideNumX-1+i*subSideNumX];
}
MPI_Send(sSendUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 9, MPI_COMM_WORLD);
MPI_Send(sSendDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 10, MPI_COMM_WORLD);
MPI_Send(sSendRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 12, MPI_COMM_WORLD);
//cout<<"SENT AT PROCESS: "<<mygrid_rank<<endl;
MPI_Recv(sRecvUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//cout<<"RECEIVD AT PROCESS: "<<mygrid_rank<<endl;
}









//--------------------------------------- Compute interior vorticity--------------------------------------- 

double siPlus = 0;
double siMinus = 0;
double sjPlus = 0;
double sjMinus = 0;
double viPlus = 0;
double viMinus = 0;
double vjPlus = 0;
double vjMinus = 0;
int nthreads, threadid;
#pragma omp parallel for default(shared) private(siPlus,siMinus,sjPlus,sjMinus,viPlus,viMinus,vjPlus,vjMinus) 

for (int i = iBegin; i < iEnd; ++i) {
for (int j = jBegin; j < jEnd; ++j) {

siPlus = sSub[subIDX(i+1,j)];
siMinus = sSub[subIDX(i-1,j)];
sjPlus = sSub[subIDX(i,j+1)];
sjMinus = sSub[subIDX(i,j-1)];

//Test if need to receive data from neigbour sub
if (i+1>subSideNum-1){
siPlus = sRecvRight[j];
}
if (i-1<0){
siMinus = sRecvLeft[j];
}
if (j+1>subSideNum-1){
sjPlus = sRecvUp[i];
}
if (j-1<0){
sjMinus = sRecvDown[i];
}


vSub[subIDX(i,j)] = dx2i*(
2.0 * sSub[subIDX(i,j)] - siPlus - siMinus)
+ 1.0/dy/dy*(
2.0 * sSub[subIDX(i,j)] - sjPlus - sjMinus);

// threadid =  omp_get_thread_num();
// cout<<"threadid = "<<threadid<<endl;
// nthreads = omp_get_num_threads();
// cout<<"nthreads = "<<nthreads<<endl;
}
} 







//Probe
// //Gather Sub v to Global at Processor 0 
//     //If not at Process 0, send u0Sub, u1Sub to Process 0
//     if (mygrid_rank != 0){
//         MPI_Send(vSub, subGridNum, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
//         //cout<<"s SENT FROM "<< mygrid_rank << " VALUE = " << sSub[7]<<endl;
//     }

//     //If at process 0
//     if (mygrid_rank == 0){
//         //cout<<"PROCESS 0 FOR V AND S BEGINS"<<endl;

//         //for All Process
//         for (int i = 0; i < size; i++) {

//             //Process 0
//             if(i==0){
//                 for (int j=0; j<subGridNum; j++){
//                    v[subIDXglobal(j,0)] = vSub[j];
//                 }
//             }

//             //At Other Processes
//             else if (i != 0){
//                 double vtemp[subGridNumArray[i]] = {};

//                 MPI_Recv(vtemp, subGridNumArray[i], MPI_DOUBLE, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//                 //cout<<"s RECEIVED FROM "<< i << " VALUE = " << stemp[7]<<endl;

//                 for (int j=0; j<subGridNumArray[i]; j++){
//                     v[subIDXglobal(j,i)] = vtemp[j];
//                 }                
//             }
//         }
//     }
//     //Gather Sub v to Global at Processor 0


//     //Probe for s
//     if (mygrid_rank==0){
//         cout<< "; s VALUE AFTER TIME INTERIOR= ";
//         for (int i=0; i<Npts; i++){
//             cout<< s[i] << ", ";
//         }
//         cout<<endl;
//     }
//     //

//     //Probe for v
//     if (mygrid_rank==0){
//         cout<< "; v VALUE AFTER TIME INTERIOR= ";
//         for (int i=0; i<Npts; i++){
//             cout<< v[i] << ", ";
//         }
//         cout<<endl;
//     }
//     //









//Communicate with Neighbour Sub Processes
//if sub not at any boundaries
if (coords[0]!=0 && coords[0]!=sideSubCoreNum-1 && coords[1]!=0 && coords[1]!=sideSubCoreNum-1){
for (int i=0; i<subSideNumX; i++){
sSendUp[i] = sSub[i+(subSideNumY-1)*subSideNumX];
sSendDown[i] = sSub[i];
vSendUp[i] = vSub[i+(subSideNumY-1)*subSideNumX];
vSendDown[i] = vSub[i];
}
for (int i=0; i<subSideNumY; i++){
sSendLeft[i] = sSub[i*subSideNumX];
sSendRight[i] = sSub[subSideNumX-1+i*subSideNumX];
vSendLeft[i] = vSub[i*subSideNumX];
vSendRight[i] = vSub[subSideNumX-1+i*subSideNumX];
}
MPI_Send(sSendUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 9, MPI_COMM_WORLD);
MPI_Send(sSendDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 10, MPI_COMM_WORLD);
MPI_Send(sSendLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 11, MPI_COMM_WORLD);
MPI_Send(sSendRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 12, MPI_COMM_WORLD);
MPI_Send(vSendUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 21, MPI_COMM_WORLD);
MPI_Send(vSendDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 22, MPI_COMM_WORLD);
MPI_Send(vSendLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 23, MPI_COMM_WORLD);
MPI_Send(vSendRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 24, MPI_COMM_WORLD);
//cout<<"SENT AT PROCESS: "<<mygrid_rank<<endl;
MPI_Recv(sRecvUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(vRecvUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 22, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(vRecvDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 21, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(vRecvLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 24, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(vRecvRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 23, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//cout<<"RECEIVD AT PROCESS: "<<mygrid_rank<<endl;
}

//if sub at bottom left corner
else if (coords[0]==0 && coords[1]==0 && sideSubCoreNum>1){
for (int i=0; i<subSideNumX; i++){
sSendUp[i] = sSub[i+(subSideNumY-1)*subSideNumX];
vSendUp[i] = vSub[i+(subSideNumY-1)*subSideNumX];
}
for (int i=0; i<subSideNumY; i++){
sSendRight[i] = sSub[subSideNumX-1+i*subSideNumX];
vSendRight[i] = vSub[subSideNumX-1+i*subSideNumX];
}
MPI_Send(sSendUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 9, MPI_COMM_WORLD);
MPI_Send(sSendRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 12, MPI_COMM_WORLD);
MPI_Send(vSendUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 21, MPI_COMM_WORLD);
MPI_Send(vSendRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 24, MPI_COMM_WORLD);
//cout<<"SENT AT PROCESS: "<<mygrid_rank<<endl;
MPI_Recv(sRecvUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(vRecvUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 22, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(vRecvRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 23, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//cout<<"RECEIVD AT PROCESS: "<<mygrid_rank<<endl;
}

//if sub at bottom right corner
else if (coords[0]==0 && coords[1]==sideSubCoreNum-1 && sideSubCoreNum>1){
for (int i=0; i<subSideNumX; i++){
sSendUp[i] = sSub[i+(subSideNumY-1)*subSideNumX];
vSendUp[i] = vSub[i+(subSideNumY-1)*subSideNumX];
}
for (int i=0; i<subSideNumY; i++){
sSendLeft[i] = sSub[i*subSideNumX];
vSendLeft[i] = vSub[i*subSideNumX];
}
MPI_Send(sSendUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 9, MPI_COMM_WORLD);
MPI_Send(sSendLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 11, MPI_COMM_WORLD);
MPI_Send(vSendUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 21, MPI_COMM_WORLD);
MPI_Send(vSendLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 23, MPI_COMM_WORLD);
//cout<<"SENT AT PROCESS: "<<mygrid_rank<<endl;
MPI_Recv(sRecvUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(vRecvUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 22, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(vRecvLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 24, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//cout<<"RECEIVD AT PROCESS: "<<mygrid_rank<<endl;
}

//if sub at top left corner
else if (coords[0]==sideSubCoreNum-1 && coords[1]==0 && sideSubCoreNum>1){
for (int i=0; i<subSideNumX; i++){
sSendDown[i] = sSub[i];
vSendDown[i] = vSub[i];
}
for (int i=0; i<subSideNumY; i++){
sSendRight[i] = sSub[subSideNumX-1+i*subSideNumX];
vSendRight[i] = vSub[subSideNumX-1+i*subSideNumX];
}
MPI_Send(sSendDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 10, MPI_COMM_WORLD);
MPI_Send(sSendRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 12, MPI_COMM_WORLD);
MPI_Send(vSendDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 22, MPI_COMM_WORLD);
MPI_Send(vSendRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 24, MPI_COMM_WORLD);
//cout<<"SENT AT PROCESS: "<<mygrid_rank<<endl;
MPI_Recv(sRecvDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(vRecvDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 21, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(vRecvRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 23, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//cout<<"RECEIVD AT PROCESS: "<<mygrid_rank<<endl;
}

//if sub at top right corner
else if (coords[0]==sideSubCoreNum-1 && coords[1]==sideSubCoreNum-1 && sideSubCoreNum>1){
for (int i=0; i<subSideNumX; i++){
sSendDown[i] = sSub[i];
vSendDown[i] = vSub[i];
}
for (int i=0; i<subSideNumY; i++){
sSendLeft[i] = sSub[i*subSideNumX];
vSendLeft[i] = vSub[i*subSideNumX];
}
MPI_Send(sSendDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 10, MPI_COMM_WORLD);
MPI_Send(sSendLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 11, MPI_COMM_WORLD);
MPI_Send(vSendDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 22, MPI_COMM_WORLD);
MPI_Send(vSendLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 23, MPI_COMM_WORLD);
//cout<<"SENT AT PROCESS: "<<mygrid_rank<<endl;
MPI_Recv(sRecvDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(vRecvDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 21, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(vRecvLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 24, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//cout<<"RECEIVD AT PROCESS: "<<mygrid_rank<<endl;
}

//if sub at top edge
else if (coords[0]==sideSubCoreNum-1 && coords[1]!=sideSubCoreNum-1 && coords[1]!=0){
for (int i=0; i<subSideNumX; i++){
sSendDown[i] = sSub[i];
vSendDown[i] = vSub[i];
}
for (int i=0; i<subSideNumY; i++){
sSendLeft[i] = sSub[i*subSideNumX];
sSendRight[i] = sSub[subSideNumX-1+i*subSideNumX];
vSendLeft[i] = vSub[i*subSideNumX];
vSendRight[i] = vSub[subSideNumX-1+i*subSideNumX];
}
MPI_Send(sSendDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 10, MPI_COMM_WORLD);
MPI_Send(sSendLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 11, MPI_COMM_WORLD);
MPI_Send(sSendRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 12, MPI_COMM_WORLD);
MPI_Send(vSendDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 22, MPI_COMM_WORLD);
MPI_Send(vSendLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 23, MPI_COMM_WORLD);
MPI_Send(vSendRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 24, MPI_COMM_WORLD);
//cout<<"SENT AT PROCESS: "<<mygrid_rank<<endl;
MPI_Recv(sRecvDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(vRecvDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 21, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(vRecvLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 24, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(vRecvRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 23, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//cout<<"RECEIVD AT PROCESS: "<<mygrid_rank<<endl;
}

//if sub at bottom edge
else if (coords[0]==0 && coords[1]!=sideSubCoreNum-1 && coords[1]!=0){
for (int i=0; i<subSideNumX; i++){
sSendUp[i] = sSub[i+(subSideNumY-1)*subSideNumX];
vSendUp[i] = vSub[i+(subSideNumY-1)*subSideNumX];
}
for (int i=0; i<subSideNumY; i++){
sSendLeft[i] = sSub[i*subSideNumX];
sSendRight[i] = sSub[subSideNumX-1+i*subSideNumX];
vSendLeft[i] = vSub[i*subSideNumX];
vSendRight[i] = vSub[subSideNumX-1+i*subSideNumX];
}
MPI_Send(sSendUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 9, MPI_COMM_WORLD);
MPI_Send(sSendLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 11, MPI_COMM_WORLD);
MPI_Send(sSendRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 12, MPI_COMM_WORLD);
MPI_Send(vSendUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 21, MPI_COMM_WORLD);
MPI_Send(vSendLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 23, MPI_COMM_WORLD);
MPI_Send(vSendRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 24, MPI_COMM_WORLD);
//cout<<"SENT AT PROCESS: "<<mygrid_rank<<endl;
MPI_Recv(sRecvUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(vRecvUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 22, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(vRecvLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 24, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(vRecvRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 23, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//cout<<"RECEIVD AT PROCESS: "<<mygrid_rank<<endl;
}

//if sub at right edge
else if (coords[1]==sideSubCoreNum-1 && coords[0]!=sideSubCoreNum-1 && coords[0]!=0){
for (int i=0; i<subSideNumX; i++){
sSendUp[i] = sSub[i+(subSideNumY-1)*subSideNumX];
sSendDown[i] = sSub[i];
vSendUp[i] = vSub[i+(subSideNumY-1)*subSideNumX];
vSendDown[i] = vSub[i];
}
for (int i=0; i<subSideNumY; i++){
sSendLeft[i] = sSub[i*subSideNumX];
vSendLeft[i] = vSub[i*subSideNumX];
}
MPI_Send(sSendUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 9, MPI_COMM_WORLD);
MPI_Send(sSendDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 10, MPI_COMM_WORLD);
MPI_Send(sSendLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 11, MPI_COMM_WORLD);
MPI_Send(vSendUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 21, MPI_COMM_WORLD);
MPI_Send(vSendDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 22, MPI_COMM_WORLD);
MPI_Send(vSendLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 23, MPI_COMM_WORLD);
//cout<<"SENT AT PROCESS: "<<mygrid_rank<<endl;
MPI_Recv(sRecvUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(vRecvUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 22, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(vRecvDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 21, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(vRecvLeft, subSideNumY, MPI_DOUBLE, mygrid_rank-1, 24, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//cout<<"RECEIVD AT PROCESS: "<<mygrid_rank<<endl;
}
//if sub at left edge
else if (coords[1]==0 && coords[0]!=sideSubCoreNum-1 && coords[0]!=0){
for (int i=0; i<subSideNumX; i++){
sSendUp[i] = sSub[i+(subSideNumY-1)*subSideNumX];
sSendDown[i] = sSub[i];
vSendUp[i] = vSub[i+(subSideNumY-1)*subSideNumX];
vSendDown[i] = vSub[i];
}
for (int i=0; i<subSideNumY; i++){
sSendRight[i] = sSub[subSideNumX-1+i*subSideNumX];
vSendRight[i] = vSub[subSideNumX-1+i*subSideNumX];
}
MPI_Send(sSendUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 9, MPI_COMM_WORLD);
MPI_Send(sSendDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 10, MPI_COMM_WORLD);
MPI_Send(sSendRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 12, MPI_COMM_WORLD);
MPI_Send(vSendUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 21, MPI_COMM_WORLD);
MPI_Send(vSendDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 22, MPI_COMM_WORLD);
MPI_Send(vSendRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 24, MPI_COMM_WORLD);
//cout<<"SENT AT PROCESS: "<<mygrid_rank<<endl;
MPI_Recv(sRecvUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(sRecvRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(vRecvUp, subSideNumX, MPI_DOUBLE, mygrid_rank+sideSubCoreNum, 22, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(vRecvDown, subSideNumX, MPI_DOUBLE, mygrid_rank-sideSubCoreNum, 21, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(vRecvRight, subSideNumY, MPI_DOUBLE, mygrid_rank+1, 23, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//cout<<"RECEIVD AT PROCESS: "<<mygrid_rank<<endl;
}


// //Probe for vSub
// if (mygrid_rank==0){
//     cout<< "vSub VALUE BEFORE TIME ADVANCE= ";
//     for (int i=0; i<subGridNum; i++){
//         cout<< vSub[i] << ", ";
//     }
//     cout<<endl;
// }
// //

double stemp3[subGridNum] = {};
double vtemp3[subGridNum] = {};

for (int i=0; i<subGridNum;i++){
stemp3[i]=sSub[i];
vtemp3[i]=vSub[i];
}


//--------------------------------------- Time advance vorticity--------------------------------------- 

#pragma omp parallel for default(shared) private(siPlus,siMinus,sjPlus,sjMinus,viPlus,viMinus,vjPlus,vjMinus) 
for (int i = iBegin; i < iEnd; ++i) {
for (int j = jBegin; j < jEnd; ++j) {

siPlus = stemp3[subIDX(i+1,j)];
siMinus = stemp3[subIDX(i-1,j)];
sjPlus = stemp3[subIDX(i,j+1)];
sjMinus = stemp3[subIDX(i,j-1)];
viPlus = vtemp3[subIDX(i+1,j)];
viMinus = vtemp3[subIDX(i-1,j)];
vjPlus = vtemp3[subIDX(i,j+1)];
vjMinus = vtemp3[subIDX(i,j-1)];

// //Probe
// if (mygrid_rank==3&&i==2&&j==3){
//     cout<<"vjPlus 1 = "<< vjPlus<<endl;
// }

//Test if need to receive data from neigbour sub
if (i+1>subSideNumX-1){
viPlus = vRecvRight[j];
siPlus = sRecvRight[j];
}
if (i-1<0){
siMinus = sRecvLeft[j];
viMinus = vRecvLeft[j];
}
if (j+1>subSideNumY-1){
sjPlus = sRecvUp[i];
vjPlus = vRecvUp[i];
}
if (j-1<0){
sjMinus = sRecvDown[i];
vjMinus = vRecvDown[i];
}

// //Probe
// if (mygrid_rank==8 && i==1 && j==1){
//     cout<<"v BEFORE = "<< vSub[subIDX(i,j)]<<endl;
//     cout<<"dx2i = "<< dx2i<<endl;
//     cout<<"dy2i = "<<dy2i<<endl;
//     cout<<"nu = "<<nu<<endl;
//     cout<<"dt = "<< dt<<endl;                      
//     cout<<"siPlus = "<< siPlus<<endl;
//     cout<<"siMinus = "<< siMinus<<endl;
//     cout<<"sjPlus = "<< sjPlus<<endl;
//     cout<<"sjMinus = "<< sjMinus<<endl;
//     cout<<"viPlus = "<< viPlus<<endl;
//     cout<<"viMinus = "<< viMinus<<endl;
//     cout<<"vjPlus = "<< vjPlus<<endl;
//     cout<<"vjMinus = "<< vjMinus<<endl;
//     cout<<"vjMinus = "<< vjMinus<<endl; 
// }

//Formula
vSub[subIDX(i,j)] = vSub[subIDX(i,j)] + dt*(
( (siPlus - siMinus) * 0.5 * dxi
*(vjPlus - vjMinus) * 0.5 * dyi)
- ( (sjPlus - sjMinus) * 0.5 * dyi
*(viPlus - viMinus) * 0.5 * dxi)
+ nu * (viPlus - 2.0 * vSub[subIDX(i,j)] + viMinus)*dx2i
+ nu * (vjPlus - 2.0 * vSub[subIDX(i,j)] + vjMinus)*dy2i);


// //Probe
// if (mygrid_rank==8 && i==1 && j==0){
//     cout<<"v = "<< vSub[subIDX(i,j)]<<endl;
// }

}
}
//--------------------------------------- Time advance vorticity End--------------------------------------- 





// //Probe for vSub
// if (mygrid_rank==6){
//     //cout<< "; vSub VALUE AFTER TIME ADVANCE= ";
//     for (int i=0; i<subGridNum; i++){
//         cout<< vSub[i] << ", ";
//     }
//     cout<<endl;
// }
// //



//Gather Sub v to Global at Processor 0 
//If not at Process 0, send u0Sub, u1Sub to Process 0
if (mygrid_rank != 0){
MPI_Send(vSub, subGridNum, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
//cout<<"s SENT FROM "<< mygrid_rank << " VALUE = " << sSub[7]<<endl;
}

//If at process 0
if (mygrid_rank == 0){
//cout<<"PROCESS 0 FOR V AND S BEGINS"<<endl;

//for All Process
for (int i = 0; i < size; i++) {

//Compute Process 0
if(i==0){

#pragma omp parallel for        
for (int j=0; j<subGridNum; j++){
v[subIDXglobal(j,0)] = vSub[j];
}
}

//Compute Other Processes
else if (i != 0){
double vtemp[subGridNumArray[i]] = {};

MPI_Recv(vtemp, subGridNumArray[i], MPI_DOUBLE, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//cout<<"s RECEIVED FROM "<< i << " VALUE = " << stemp[7]<<endl;

#pragma omp parallel for        
for (int j=0; j<subGridNumArray[i]; j++){
v[subIDXglobal(j,i)] = vtemp[j];
}                
}
}

}
//Gather Sub v to Global at Processor 0


//Broadcast v s from proecess 0
if (mygrid_rank==0){

double vtemp3[Npts]={};
double stemp3[Npts]={};

for (int j=0; j<Npts; j++){
vtemp3[j]=v[j];
stemp3[j]=s[j];
}

for (int i=1; i<size; i++){
MPI_Send(vtemp3, Npts, MPI_DOUBLE, i, 13, MPI_COMM_WORLD);
MPI_Send(stemp3, Npts, MPI_DOUBLE, i, 14, MPI_COMM_WORLD);
}
}
//At processes other than 0
else{
double vtemp3[Npts]={};
double stemp3[Npts]={};
MPI_Recv(vtemp3, Npts, MPI_DOUBLE, 0, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(stemp3, Npts, MPI_DOUBLE, 0, 14, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

for (int j=0; j<Npts; j++){
v[j]=vtemp3[j];
s[j]=stemp3[j];
}     
}

// //Probe for s
// if (mygrid_rank==0){
//     cout<< "; s VALUE AFTER TIME ADVANCE= ";
//     for (int i=0; i<Npts; i++){
//         cout<< s[i] << ", ";
//     }
//     cout<<endl;
// }
// //

// //Probe for v
// if (mygrid_rank==0){
//     cout<< "; v VALUE AFTER TIME ADVANCE= ";
//     for (int i=0; i<Npts; i++){
//         cout<< v[i] << ", ";
//     }
//     cout<<endl;
// }
// //



// Sinusoidal test case with analytical solution, which can be used to test
// the Poisson solver
/*
const int k = 3;
const int l = 3;
for (int i = 0; i < Nx; ++i) {
for (int j = 0; j < Ny; ++j) {
v[IDX(i,j)] = -M_PI * M_PI * (k * k + l * l)
* sin(M_PI * k * i * dx)
* sin(M_PI * l * j * dy);
}
}
*/





// //Probe for s
// if (mygrid_rank==0){
//     cout<< "; s VALUE BEFORE SENT TO SOLVERCG= ";
//     for (int i=0; i<Npts; i++){
//         cout<< s[i] << ", ";
//     }
//     cout<<endl;
// }
// //

// //Probe for v
// if (mygrid_rank==0){
//     cout<< "; v VALUE BEFORE SENT TO SOLVERCG= ";
//     for (int i=0; i<Npts; i++){
//         cout<< v[i] << ", ";  
//     }
//     cout<<endl;
// }
// //


// Solve Poisson problem
cg->Solve(v, s);




// //Probe for s
// if (mygrid_rank==0){
//     cout<< "; s VALUE AFTER SOLVERCG= ";
//     for (int i=0; i<Npts; i++){
//         cout<< s[i] << ", ";
//     }
//     cout<<endl;
// }
// //


// //Probe for v
// if (mygrid_rank==0){
//     cout<< "; v VALUE received from SOLVERCG= ";
//     for (int i=0; i<Npts; i++){
//         cout<< v[i] << ", ";  
//     }
//     cout<<endl;
// }
// //

if (mygrid_rank==0){

double vtemp2[Npts]={};
double stemp2[Npts]={};

#pragma omp parallel for
for (int j=0; j<Npts; j++){
vtemp2[j]=v[j];
stemp2[j]=s[j];
}

for (int i=1; i<size; i++){
MPI_Send(vtemp2, Npts, MPI_DOUBLE, i, 13, MPI_COMM_WORLD);
MPI_Send(stemp2, Npts, MPI_DOUBLE, i, 14, MPI_COMM_WORLD);
}
}
//At processes other than 0
else{
double vtemp2[Npts]={};
double stemp2[Npts]={};
MPI_Recv(vtemp2, Npts, MPI_DOUBLE, 0, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(stemp2, Npts, MPI_DOUBLE, 0, 14, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

#pragma omp parallel for
for (int j=0; j<Npts; j++){
v[j]=vtemp2[j];
s[j]=stemp2[j];
// threadid =  omp_get_thread_num();
// cout<<"threadid = "<<threadid<<endl;
// nthreads = omp_get_num_threads();
// cout<<"nthreads = "<<nthreads<<endl;
}     
}


}
