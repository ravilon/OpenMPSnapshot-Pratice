#include <iostream>
#include <algorithm>
#include <cstring>
using namespace std;

#include <cblas.h>
#include <mpi.h>
#include <omp.h>

#include "SolverCG.h"

//sub coords to sub index (Checked)
#define subIDX(i,j) ((j)*subSideNumX + (i))

#define IDX(I,J) ((J)*Nx + (I))

SolverCG::SolverCG(int pNx, int pNy, double pdx, double pdy)
{
dx = pdx;
dy = pdy;
Nx = pNx;
Ny = pNy;
int n = Nx*Ny;
r = new double[n];
p = new double[n];
z = new double[n];
t = new double[n]; //temp
}


//convert sub index to global index (CHECKED)
int SolverCG::subIDXglobal(int iSub, int rank)
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


SolverCG::~SolverCG()
{
delete[] r;
delete[] p;
delete[] z;
delete[] t;
}

//Function to visualise values in the do while loop
void SolverCG::probe(int K, int RANK, int k, int rank, double* x, double* t, double* p, int n, double* z, double alpha, double beta) {
if (k==K && rank==RANK){
//Probe 
cout<< "x=s:Streamfuction VALUE = ";
for (int i=0; i<n; i++){
cout<< x[i] << ", ";
}
cout<<endl;
//Probe
cout<< "t=v:Vorticity VALUE 2= ";
for (int i=0; i<n; i++){
cout<< t[i] << ", ";
}
cout<<endl;   
//Probe 
cout<< "p VALUE = "<< *p << endl;
//Probe 
cout<< "n VALUE = "<< n << endl;
//Probe 
cout<< "r VALUE 2= ";
for (int i=0; i<n; i++){
cout<< r[i] << ", ";
}
cout<<endl;    
//Probe 
cout<< "z VALUE = "<< *z << endl;
//Probe for alpha and beta 
cout<< "alpha VALUE = "<< alpha << endl;
cout<< "beta VALUE = "<< beta << endl;
}        
}




void SolverCG::Solve(double* b, double* x) {//b=v vorticity, x=s streamfunction 


//Get size and rank
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);
//cout << "Process Number = "<< rank+1<< " of " << size << endl;

//num of process per each side
sideSubCoreNum = sqrt(size);

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




unsigned int n = Nx*Ny;//Total Grid Num
int k;  //Num of Iteration
double alpha=0.0;
double beta=0.0;
double eps;//eps is the absolute squre root of b
double tol = 0.001;


// //Probe for alpha and beta 
// if (rank==0){
//     cout<< "alpha VALUE = "<< alpha << endl;
//     cout<< "beta VALUE = "<< beta << endl;
// }


//test if converge
eps = cblas_dnrm2(n, b, 1);//eps = root(b1^2 + b2^2 + b3^2 +...)
if (eps < tol*tol) {
std::fill(x, x+n, 0.0);//将解向量x中的所有元素置为零
cout << "Norm is " << eps << " at Rank "<<rank<< endl;
return;
}

// cout<<"n = "<<n<<endl;

//  //Probe 
//     if (rank==0){
//         cout<< "x=s:Streamfuction VALUE = ";
//         for (int i=0; i<n; i++){
//             cout<< x[i] << ", ";
//         }
//         cout<<endl;
//     }
//     //Probe
//     if (rank==0){
//         cout<< "t=v:Vorticity VALUE = ";
//         for (int i=0; i<n; i++){
//             cout<< t[i] << ", ";
//         }
//         cout<<endl;
//     }
//     //Probe 
//     if (rank==0){
//         cout<< "p VALUE = "<< *p << endl;
//     }    
//     //Probe 
//     if (rank==0){
//         cout<< "n VALUE = "<< n << endl;
//     }
//     //Probe 
//     if (rank==0){
//         cout<< "r VALUE = ";
//         for (int i=0; i<n; i++){
//             cout<< r[i] << ", ";
//         }
//         cout<<endl;
//     }
//     //Probe 
//     if (rank==0){
//         cout<< "z VALUE = "<< *z << endl;
//     }
//     //Probe for alpha and beta 
//     if (rank==0){
//         cout<< "alpha VALUE = "<< alpha << endl;
//         cout<< "beta VALUE = "<< beta << endl;
//     }





ApplyOperator(x, t);//x=s:streamfunction, t=vorticity


// //Probe 
// if (rank==0){
//     cout<< "x=s:Streamfuction VALUE = ";
//     for (int i=0; i<n; i++){
//         cout<< x[i] << ", ";
//     }
//     cout<<endl;
// }
// //Probe
// if (rank==0){
//     cout<< "t=v:Vorticity VALUE = ";
//     for (int i=0; i<n; i++){
//         cout<< t[i] << ", ";
//     }
//     cout<<endl;
// }
// //Probe 
// if (rank==0){
//     cout<< "p VALUE = "<< *p << endl;
// }    
// //Probe 
// if (rank==0){
//     cout<< "n VALUE = "<< n << endl;
// }
// //Probe 
// if (rank==0){
//     cout<< "r VALUE = ";
//     for (int i=0; i<n; i++){
//         cout<< r[i] << ", ";
//     }
//     cout<<endl;
// }
// //Probe 
// if (rank==0){
//     cout<< "z VALUE = "<< *z << endl;
// }
// //Probe for alpha and beta 
// if (rank==0){
//     cout<< "alpha VALUE = "<< alpha << endl;
//     cout<< "beta VALUE = "<< beta << endl;
// }



cblas_dcopy(n, b, 1, r, 1);        // r_0 = b (i.e. b)



if (rank==0){
// int nthreads, threadid;  

#pragma omp parallel for 
// Boundaries
for (int i = 0; i < Nx; ++i) {
r[IDX(i, 0)] = 0.0;
r[IDX(i, Ny-1)] = 0.0;
// threadid=omp_get_thread_num();
// cout<<"threadid = "<<threadid<<endl;
}

#pragma omp parallel for 
for (int j = 0; j < Ny; ++j) {
r[IDX(0, j)] = 0.0;
r[IDX(Nx - 1, j)] = 0.0;
}
}


cblas_daxpy(n, -1.0, t, 1, r, 1); //r = -t+r





//--------------------------Precondition BEGIN---------
int i, j;
double dx2i = 1.0/dx/dx;
double dy2i = 1.0/dy/dy;
double factor = 2.0*(dx2i + dy2i);


//
double intemp[n]={};
double outtemp[n]={};
if (rank==0){
for (int j=0; j<n; j++){
intemp[j]=r[j];
outtemp[j]=z[j];
}
for (int i=1; i<size; i++){
MPI_Send(intemp, n, MPI_DOUBLE, i, 33, MPI_COMM_WORLD);
MPI_Send(outtemp, n, MPI_DOUBLE, i, 34, MPI_COMM_WORLD);
}
//cout<<"PROCESS 0 SENT 1"<<endl;
}
else if (rank != 0){
//cout<<"PROCESS "<<mygrid_rank<<"Begines RECEIVED"<<endl;
MPI_Recv(intemp, n, MPI_DOUBLE, 0, 33, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(outtemp, n, MPI_DOUBLE, 0, 34, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


for (int j=0; j<n; j++){
r[j]=intemp[j];
z[j]=outtemp[j];
}     
//cout<<"PROCESS "<<mygrid_rank<<"RECEIVED"<<endl;
}


// //Probe for intemp
// if (rank==0){
//     cout<< "in VALUE at rank 2 = ";
//     for (int i=0; i<n; i++){
//         cout<< in[i] << ", ";
//     }
//     cout<<endl;
// }
// //Probe for outtemp
// if (rank==0){
//     cout<< "out VALUE = ";
//     for (int i=0; i<n; i++){
//         cout<< out[i] << ", ";
//     }
//     cout<<endl;
// }


//Initialize v and s for Sub Cores
double inSub[subGridNum] = {};
double outSub[subGridNum] = {};

//Convert Global v, s to Sub vSub, sSub
for (int i=0;i<subGridNum;i++){
inSub[i] = r[subIDXglobal(i, rank)];
outSub[i] = z[subIDXglobal(i, rank)];
}   


// //Probe
// if (rank==7){
//     cout<< "inSub VALUE = ";
//     for (int i=0; i<subGridNum; i++){
//         cout<< inSub[i] << ", ";
//     }
//     cout<<endl;
//     cout<< "outSub VALUE = ";
//     for (int i=0; i<subGridNum; i++){
//         cout<< outSub[i] << ", ";
//     }
//     cout<<endl;
// }//test sub position when alogrithm nelegecting boundary
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

// Boundaries, let out = in 
for (i = 0; i < subSideNumX; ++i) {
outSub[subIDX(i, 0)] = inSub[subIDX(i,0)];
//            outSub[subIDX(i, subSideNumY-1)] = inSub[subIDX(i, subSideNumY-1)];
}
for (j = 0; j < subSideNumY; ++j) {
outSub[subIDX(0, j)] = inSub[subIDX(0, j)];
//            outSub[subIDX(subSideNumX - 1, j)] = inSub[subIDX(subSideNumX - 1, j)];
}
}
//if sub at bottom right corner
else if (coords[0]==0 && coords[1]==sideSubCoreNum-1){
iBegin = 0;
iEnd = subSideNumX-1;
jBegin = 1;
jEnd = subSideNumY; 

// Boundaries, let out = in 
for (i = 0; i < subSideNumX; ++i) {
outSub[subIDX(i, 0)] = inSub[subIDX(i,0)];
//            outSub[subIDX(i, subSideNumY-1)] = inSub[subIDX(i, subSideNumY-1)];
}
for (j = 0; j < subSideNumY; ++j) {
//            outSub[subIDX(0, j)] = inSub[subIDX(0, j)];
outSub[subIDX(subSideNumX - 1, j)] = inSub[subIDX(subSideNumX - 1, j)];
}
}
//if sub at top left corner
else if (coords[0]==sideSubCoreNum-1 && coords[1]==0){
iBegin = 1;
iEnd = subSideNumX;
jBegin = 0;
jEnd = subSideNumY-1; 

// Boundaries, let out = in 
for (i = 0; i < subSideNumX; ++i) {
//            outSub[subIDX(i, 0)] = inSub[subIDX(i,0)];
outSub[subIDX(i, subSideNumY-1)] = inSub[subIDX(i, subSideNumY-1)];
}
for (j = 0; j < subSideNumY; ++j) {
outSub[subIDX(0, j)] = inSub[subIDX(0, j)];
//            outSub[subIDX(subSideNumX - 1, j)] = inSub[subIDX(subSideNumX - 1, j)];
}
}
//if sub at top right corner
else if (coords[0]==sideSubCoreNum-1 && coords[1]==sideSubCoreNum-1){
iBegin = 0;
iEnd = subSideNumX-1;
jBegin = 0;
jEnd = subSideNumY-1; 

// Boundaries, let out = in 
for (i = 0; i < subSideNumX; ++i) {
//            outSub[subIDX(i, 0)] = inSub[subIDX(i,0)];
outSub[subIDX(i, subSideNumY-1)] = inSub[subIDX(i, subSideNumY-1)];
}
for (j = 0; j < subSideNumY; ++j) {
//            outSub[subIDX(0, j)] = inSub[subIDX(0, j)];
outSub[subIDX(subSideNumX - 1, j)] = inSub[subIDX(subSideNumX - 1, j)];
}
}
//if sub at top edge
else if (coords[0]==sideSubCoreNum-1 && coords[1]!=sideSubCoreNum-1 && coords[1]!=0){
iBegin = 0;
iEnd = subSideNumX;
jBegin = 0;
jEnd = subSideNumY-1; 

// Boundaries, let out = in 
for (i = 0; i < subSideNumX; ++i) {
//            outSub[subIDX(i, 0)] = inSub[subIDX(i,0)];
outSub[subIDX(i, subSideNumY-1)] = inSub[subIDX(i, subSideNumY-1)];
}
for (j = 0; j < subSideNumY; ++j) {
//            outSub[subIDX(0, j)] = inSub[subIDX(0, j)];
//            outSub[subIDX(subSideNumX - 1, j)] = inSub[subIDX(subSideNumX - 1, j)];
}
}
//if sub at bottom edge
else if (coords[0]==0 && coords[1]!=sideSubCoreNum-1 && coords[1]!=0){
iBegin = 0;
iEnd = subSideNumX;
jBegin = 1;
jEnd = subSideNumY; 

// Boundaries, let out = in 
for (i = 0; i < subSideNumX; ++i) {
outSub[subIDX(i, 0)] = inSub[subIDX(i,0)];
//            outSub[subIDX(i, subSideNumY-1)] = inSub[subIDX(i, subSideNumY-1)];
}
for (j = 0; j < subSideNumY; ++j) {
//            outSub[subIDX(0, j)] = inSub[subIDX(0, j)];
//            outSub[subIDX(subSideNumX - 1, j)] = inSub[subIDX(subSideNumX - 1, j)];
}
}
//if sub at right edge
else if (coords[1]==sideSubCoreNum-1 && coords[0]!=sideSubCoreNum-1 && coords[0]!=0){
iBegin = 0;
iEnd = subSideNumX-1;
jBegin = 0;
jEnd = subSideNumY; 

// Boundaries, let out = in 
for (i = 0; i < subSideNumX; ++i) {
//            outSub[subIDX(i, 0)] = inSub[subIDX(i,0)];
//            outSub[subIDX(i, subSideNumY-1)] = inSub[subIDX(i, subSideNumY-1)];
}
for (j = 0; j < subSideNumY; ++j) {
//            outSub[subIDX(0, j)] = inSub[subIDX(0, j)];
outSub[subIDX(subSideNumX - 1, j)] = inSub[subIDX(subSideNumX - 1, j)];
}
}
//if sub at left edge
else if (coords[1]==0 && coords[0]!=sideSubCoreNum-1 && coords[0]!=0){
iBegin = 1;
iEnd = subSideNumX;
jBegin = 0;
jEnd = subSideNumY; 
// Boundaries, let out = in 
for (i = 0; i < subSideNumX; ++i) {
//            outSub[subIDX(i, 0)] = inSub[subIDX(i,0)];
//            outSub[subIDX(i, subSideNumY-1)] = inSub[subIDX(i, subSideNumY-1)];
}
for (j = 0; j < subSideNumY; ++j) {
outSub[subIDX(0, j)] = inSub[subIDX(0, j)];
//            outSub[subIDX(subSideNumX - 1, j)] = inSub[subIDX(subSideNumX - 1, j)];
}
}


// int nthreads, threadid;
#pragma omp parallel for 
//For interior, let out = in / factor
for (int j = jBegin; j < jEnd; ++j) 
{
for (int i = iBegin; i < iEnd; ++i) {
outSub[subIDX(i,j)] = inSub[subIDX(i,j)]/factor;
// threadid =  omp_get_thread_num();
// cout<<"threadid = "<<threadid<<endl;
// nthreads = omp_get_num_threads();
// cout<<"nthreads = "<<nthreads<<endl;
}
}


//Gather Sub v to Global at Processor 0 
//If not at Process 0, send u0Sub, u1Sub to Process 0
if (mygrid_rank != 0){
MPI_Send(inSub, subGridNum, MPI_DOUBLE, 0, 233, MPI_COMM_WORLD);
MPI_Send(outSub, subGridNum, MPI_DOUBLE, 0, 243, MPI_COMM_WORLD);
//cout<<"s SENT FROM "<< mygrid_rank << " VALUE = " << sSub[7]<<endl;
}

//If at process 0
if (mygrid_rank == 0){
//cout<<"PROCESS 0 FOR V AND S BEGINS"<<endl;

//for All Process
for (int i = 0; i < size; i++) {

//Process 0
if(i==0){
#pragma omp parallel for        
for (int j=0; j<subGridNum; j++){
r[subIDXglobal(j,0)] = inSub[j];                   
z[subIDXglobal(j,0)] = outSub[j];     
}
}

//At Other Processes
else if (i != 0){
double intemp1[subGridNumArray[i]] = {};
double outtemp1[subGridNumArray[i]] = {};

MPI_Recv(intemp1, subGridNumArray[i], MPI_DOUBLE, i, 233, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(outtemp1, subGridNumArray[i], MPI_DOUBLE, i, 243, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//cout<<"s RECEIVED FROM "<< i << " VALUE = " << stemp[7]<<endl;

#pragma omp parallel for        
for (int j=0; j<subGridNumArray[i]; j++){
r[subIDXglobal(j,i)] = intemp1[j];
z[subIDXglobal(j,i)] = outtemp1[j];
}                
}
}

}
//Gather Sub v to Global at Processor 0



//Broadcast v s from proecess 0
if (mygrid_rank==0){

//     for (int j = 1; j < Ny - 1; ++j) {
//     for (int i = 1; i < Nx - 1; ++i) {
//         out[IDX(i,j)] = ( -     in[IDX(i-1, j)]
//                           + 2.0*in[IDX(i,   j)]
//                           -     in[IDX(i+1, j)])*dx2i
//                       + ( -     in[IDX(i, j-1)]
//                           + 2.0*in[IDX(i,   j)]
//                           -     in[IDX(i, j+1)])*dy2i;

//     }
//     jm1++;
//     jp1++;
// }

double intemp2[n]={};
double outtemp2[n]={};

for (int j=0; j<n; j++){
intemp2[j]=r[j];
outtemp2[j]=z[j];
}

for (int i=1; i<size; i++){
MPI_Send(intemp2, n, MPI_DOUBLE, i, 513, MPI_COMM_WORLD);
MPI_Send(outtemp2, n, MPI_DOUBLE, i, 514, MPI_COMM_WORLD);
}
}
//At processes other than 0
else{
double intemp2[n]={};
double outtemp2[n]={};
MPI_Recv(intemp2, n, MPI_DOUBLE, 0, 513, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(outtemp2, n, MPI_DOUBLE, 0, 514, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

for (int j=0; j<n; j++){
r[j]=intemp2[j];
z[j]=outtemp2[j];
}     
}


//--------------------------Precondition END-----------






cblas_dcopy(n, z, 1, p, 1);        // p_0 = r_0


// //Probe 
// if (rank==0){
//     cout<< "x=s:Streamfuction VALUE = ";
//     for (int i=0; i<n; i++){
//         cout<< x[i] << ", ";
//     }
//     cout<<endl;
// }
// //Probe
// if (rank==0){
//     cout<< "t=v:Vorticity VALUE = ";
//     for (int i=0; i<n; i++){
//         cout<< t[i] << ", ";
//     }
//     cout<<endl;
// }
// //Probe 
// if (rank==0){
//     cout<< "p VALUE = "<< *p << endl;
// }    
// //Probe 
// if (rank==0){
//     cout<< "n VALUE = "<< n << endl;
// }
// //Probe 
// if (rank==0){
//     cout<< "r VALUE = ";
//     for (int i=0; i<n; i++){
//         cout<< r[i] << ", ";
//     }
//     cout<<endl;
// }
// //Probe 
// if (rank==0){
//     cout<< "z VALUE = "<< *z << endl;
// }
// //Probe for alpha and beta 
// if (rank==0){
//     cout<< "alpha VALUE = "<< alpha << endl;
//     cout<< "beta VALUE = "<< beta << endl;
// }






k = 0;
do {
k++;
// //(k, rank)
// probe(1, 0, k, rank, x, t, p, n, z, alpha, beta);

// Perform action of Nabla^2 * p
ApplyOperator(p, t);//p=streamfunction, t=vorticity


//Initialize v and s for Sub Cores
double pSub[subGridNum] = {};
double tSub[subGridNum] = {};

//Convert Global v, s to Sub vSub, sSub
for (int i=0;i<subGridNum;i++){
pSub[i] = p[subIDXglobal(i, rank)];
tSub[i] = t[subIDXglobal(i, rank)];
} 

//alpha = cblas_ddot(n, t, 1, p, 1);  //alpha=t^T * p                         // alpha = p_k^T A p_k
double alphaSub=0.0;
for (int i = 0; i < subGridNum; ++i) {
alphaSub += tSub[i] * pSub[i];
}
MPI_Allreduce(&alphaSub, &alpha, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

// //(k, rank)
// probe(2, 0, k, rank, x, t, p, n, z, alpha, beta);


//Initialize v and s for Sub Cores
double rSub[subGridNum] = {};
double zSub[subGridNum] = {};

//Convert Global v, s to Sub vSub, sSub
for (int i=0;i<subGridNum;i++){
rSub[i] = r[subIDXglobal(i, rank)];
zSub[i] = z[subIDXglobal(i, rank)];
} 
// alpha = cblas_ddot(n, r, 1, z, 1) / alpha; // alpha=r^T * z / (t^T * p)     // compute alpha_k
double dot_productSub = 0.0;
double dot_product = 0.0;
for (int i = 0; i < subGridNum; ++i) {
dot_productSub += rSub[i] * zSub[i];
}
MPI_Allreduce(&dot_productSub, &dot_product, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
alpha = dot_product / alpha;



//alpha = cblas_ddot(n, t, 1, p, 1);  //alpha=t^T * p                         // alpha = p_k^T A p_k

double betaSub=0.0;
//beta  = cblas_ddot(n, r, 1, z, 1); //beta=r^T * z                            // z_k^T r_k
for (int i = 0; i < subGridNum; ++i) {
betaSub += rSub[i] * zSub[i];
}
MPI_Allreduce(&betaSub, &beta, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

#pragma omp parallel for 
//cblas_daxpy(n,  alpha, p, 1, x, 1); //x= r^T * z / (t^T * p) * p + x               // x_{k+1} = x_k + alpha_k p_k
for (int i = 0; i < n; ++i) {
x[i] += alpha * p[i];
}


#pragma omp parallel for 
//cblas_daxpy(n, -alpha, t, 1, r, 1); //r=-r^T * z / (t^T * p) * t + r                // r_{k+1} = r_k - alpha_k A p_k
for (int i = 0; i < n; ++i) {
r[i] += -alpha * t[i];
}


// eps = cblas_dnrm2(n, r, 1);//eps = root(r1^2 + r2^2 + r3^2 +...)
double sum = 0.0;
#pragma omp parallel for  reduction(+:sum)
for (int i = 0; i < n; ++i) {
sum += r[i] * r[i];
}

eps = sqrt(sum);

// //(k, rank)
// probe(2, 1, k, rank, x, t, p, n, z, alpha, beta);

// if (k==2 && rank==1){
//     //Probe 
//     cout<< "eps = "<< eps << endl;
// }


if (eps < tol*tol) {
break;
}


//------------PreCondition()
int i, j;



double dx2i = 1.0/dx/dx;
double dy2i = 1.0/dy/dy;
double factor = 2.0*(dx2i + dy2i);


//
double intemp[n]={};
double outtemp[n]={};
if (rank==0){
for (int j=0; j<n; j++){
intemp[j]=r[j];
outtemp[j]=z[j];
}
for (int i=1; i<size; i++){
MPI_Send(intemp, n, MPI_DOUBLE, i, 33, MPI_COMM_WORLD);
MPI_Send(outtemp, n, MPI_DOUBLE, i, 34, MPI_COMM_WORLD);
}
//cout<<"PROCESS 0 SENT 1"<<endl;
}
else if (rank != 0){
//cout<<"PROCESS "<<mygrid_rank<<"Begines RECEIVED"<<endl;
MPI_Recv(intemp, n, MPI_DOUBLE, 0, 33, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(outtemp, n, MPI_DOUBLE, 0, 34, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


for (int j=0; j<n; j++){
r[j]=intemp[j];
z[j]=outtemp[j];
}     
//cout<<"PROCESS "<<mygrid_rank<<"RECEIVED"<<endl;
}


// //Probe for intemp
// if (rank==0){
//     cout<< "in VALUE at rank 2 = ";
//     for (int i=0; i<n; i++){
//         cout<< in[i] << ", ";
//     }
//     cout<<endl;
// }
// //Probe for outtemp
// if (rank==0){
//     cout<< "out VALUE = ";
//     for (int i=0; i<n; i++){
//         cout<< out[i] << ", ";
//     }
//     cout<<endl;
// }


//Initialize v and s for Sub Cores
double inSub[subGridNum] = {};
double outSub[subGridNum] = {};

//Convert Global v, s to Sub vSub, sSub
for (int i=0;i<subGridNum;i++){
inSub[i] = r[subIDXglobal(i, rank)];
outSub[i] = z[subIDXglobal(i, rank)];
}   


// //Probe
// if (rank==7){
//     cout<< "inSub VALUE = ";
//     for (int i=0; i<subGridNum; i++){
//         cout<< inSub[i] << ", ";
//     }
//     cout<<endl;
//     cout<< "outSub VALUE = ";
//     for (int i=0; i<subGridNum; i++){
//         cout<< outSub[i] << ", ";
//     }
//     cout<<endl;
// }//test sub position when alogrithm nelegecting boundary
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

// Boundaries, let out = in 
for (i = 0; i < subSideNumX; ++i) {
outSub[subIDX(i, 0)] = inSub[subIDX(i,0)];
//            outSub[subIDX(i, subSideNumY-1)] = inSub[subIDX(i, subSideNumY-1)];
}
for (j = 0; j < subSideNumY; ++j) {
outSub[subIDX(0, j)] = inSub[subIDX(0, j)];
//            outSub[subIDX(subSideNumX - 1, j)] = inSub[subIDX(subSideNumX - 1, j)];
}
}
//if sub at bottom right corner
else if (coords[0]==0 && coords[1]==sideSubCoreNum-1){
iBegin = 0;
iEnd = subSideNumX-1;
jBegin = 1;
jEnd = subSideNumY; 

// Boundaries, let out = in 
for (i = 0; i < subSideNumX; ++i) {
outSub[subIDX(i, 0)] = inSub[subIDX(i,0)];
//            outSub[subIDX(i, subSideNumY-1)] = inSub[subIDX(i, subSideNumY-1)];
}
for (j = 0; j < subSideNumY; ++j) {
//            outSub[subIDX(0, j)] = inSub[subIDX(0, j)];
outSub[subIDX(subSideNumX - 1, j)] = inSub[subIDX(subSideNumX - 1, j)];
}
}
//if sub at top left corner
else if (coords[0]==sideSubCoreNum-1 && coords[1]==0){
iBegin = 1;
iEnd = subSideNumX;
jBegin = 0;
jEnd = subSideNumY-1; 

// Boundaries, let out = in 
for (i = 0; i < subSideNumX; ++i) {
//            outSub[subIDX(i, 0)] = inSub[subIDX(i,0)];
outSub[subIDX(i, subSideNumY-1)] = inSub[subIDX(i, subSideNumY-1)];
}
for (j = 0; j < subSideNumY; ++j) {
outSub[subIDX(0, j)] = inSub[subIDX(0, j)];
//            outSub[subIDX(subSideNumX - 1, j)] = inSub[subIDX(subSideNumX - 1, j)];
}
}
//if sub at top right corner
else if (coords[0]==sideSubCoreNum-1 && coords[1]==sideSubCoreNum-1){
iBegin = 0;
iEnd = subSideNumX-1;
jBegin = 0;
jEnd = subSideNumY-1; 

// Boundaries, let out = in 
for (i = 0; i < subSideNumX; ++i) {
//            outSub[subIDX(i, 0)] = inSub[subIDX(i,0)];
outSub[subIDX(i, subSideNumY-1)] = inSub[subIDX(i, subSideNumY-1)];
}
for (j = 0; j < subSideNumY; ++j) {
//            outSub[subIDX(0, j)] = inSub[subIDX(0, j)];
outSub[subIDX(subSideNumX - 1, j)] = inSub[subIDX(subSideNumX - 1, j)];
}
}
//if sub at top edge
else if (coords[0]==sideSubCoreNum-1 && coords[1]!=sideSubCoreNum-1 && coords[1]!=0){
iBegin = 0;
iEnd = subSideNumX;
jBegin = 0;
jEnd = subSideNumY-1; 

// Boundaries, let out = in 
for (i = 0; i < subSideNumX; ++i) {
//            outSub[subIDX(i, 0)] = inSub[subIDX(i,0)];
outSub[subIDX(i, subSideNumY-1)] = inSub[subIDX(i, subSideNumY-1)];
}
for (j = 0; j < subSideNumY; ++j) {
//            outSub[subIDX(0, j)] = inSub[subIDX(0, j)];
//            outSub[subIDX(subSideNumX - 1, j)] = inSub[subIDX(subSideNumX - 1, j)];
}
}
//if sub at bottom edge
else if (coords[0]==0 && coords[1]!=sideSubCoreNum-1 && coords[1]!=0){
iBegin = 0;
iEnd = subSideNumX;
jBegin = 1;
jEnd = subSideNumY; 

// Boundaries, let out = in 
for (i = 0; i < subSideNumX; ++i) {
outSub[subIDX(i, 0)] = inSub[subIDX(i,0)];
//            outSub[subIDX(i, subSideNumY-1)] = inSub[subIDX(i, subSideNumY-1)];
}
for (j = 0; j < subSideNumY; ++j) {
//            outSub[subIDX(0, j)] = inSub[subIDX(0, j)];
//            outSub[subIDX(subSideNumX - 1, j)] = inSub[subIDX(subSideNumX - 1, j)];
}
}
//if sub at right edge
else if (coords[1]==sideSubCoreNum-1 && coords[0]!=sideSubCoreNum-1 && coords[0]!=0){
iBegin = 0;
iEnd = subSideNumX-1;
jBegin = 0;
jEnd = subSideNumY; 

// Boundaries, let out = in 
for (i = 0; i < subSideNumX; ++i) {
//            outSub[subIDX(i, 0)] = inSub[subIDX(i,0)];
//            outSub[subIDX(i, subSideNumY-1)] = inSub[subIDX(i, subSideNumY-1)];
}
for (j = 0; j < subSideNumY; ++j) {
//            outSub[subIDX(0, j)] = inSub[subIDX(0, j)];
outSub[subIDX(subSideNumX - 1, j)] = inSub[subIDX(subSideNumX - 1, j)];
}
}
//if sub at left edge
else if (coords[1]==0 && coords[0]!=sideSubCoreNum-1 && coords[0]!=0){
iBegin = 1;
iEnd = subSideNumX;
jBegin = 0;
jEnd = subSideNumY; 
// Boundaries, let out = in 
for (i = 0; i < subSideNumX; ++i) {
//            outSub[subIDX(i, 0)] = inSub[subIDX(i,0)];
//            outSub[subIDX(i, subSideNumY-1)] = inSub[subIDX(i, subSideNumY-1)];
}
for (j = 0; j < subSideNumY; ++j) {
outSub[subIDX(0, j)] = inSub[subIDX(0, j)];
//            outSub[subIDX(subSideNumX - 1, j)] = inSub[subIDX(subSideNumX - 1, j)];
}
}


// int nthreads, threadid;
#pragma omp parallel for 
//For interior, let out = in / factor
for (int j = jBegin; j < jEnd; ++j) 
{
for (int i = iBegin; i < iEnd; ++i) {
outSub[subIDX(i,j)] = inSub[subIDX(i,j)]/factor;
// threadid =  omp_get_thread_num();
// cout<<"threadid = "<<threadid<<endl;
// nthreads = omp_get_num_threads();
// cout<<"nthreads = "<<nthreads<<endl;
}
}


//Gather Sub v to Global at Processor 0 
//If not at Process 0, send u0Sub, u1Sub to Process 0
if (mygrid_rank != 0){
MPI_Send(inSub, subGridNum, MPI_DOUBLE, 0, 233, MPI_COMM_WORLD);
MPI_Send(outSub, subGridNum, MPI_DOUBLE, 0, 243, MPI_COMM_WORLD);
//cout<<"s SENT FROM "<< mygrid_rank << " VALUE = " << sSub[7]<<endl;
}

//If at process 0
if (mygrid_rank == 0){
//cout<<"PROCESS 0 FOR V AND S BEGINS"<<endl;

//for All Process
for (int i = 0; i < size; i++) {

//Process 0
if(i==0){
#pragma omp parallel for        
for (int j=0; j<subGridNum; j++){
r[subIDXglobal(j,0)] = inSub[j];                   
z[subIDXglobal(j,0)] = outSub[j];     
}
}

//At Other Processes
else if (i != 0){
double intemp1[subGridNumArray[i]] = {};
double outtemp1[subGridNumArray[i]] = {};

MPI_Recv(intemp1, subGridNumArray[i], MPI_DOUBLE, i, 233, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(outtemp1, subGridNumArray[i], MPI_DOUBLE, i, 243, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//cout<<"s RECEIVED FROM "<< i << " VALUE = " << stemp[7]<<endl;

#pragma omp parallel for        
for (int j=0; j<subGridNumArray[i]; j++){
r[subIDXglobal(j,i)] = intemp1[j];
z[subIDXglobal(j,i)] = outtemp1[j];
}                
}
}

}
//Gather Sub v to Global at Processor 0



//Broadcast v s from proecess 0
if (mygrid_rank==0){

//     for (int j = 1; j < Ny - 1; ++j) {
//     for (int i = 1; i < Nx - 1; ++i) {
//         out[IDX(i,j)] = ( -     in[IDX(i-1, j)]
//                           + 2.0*in[IDX(i,   j)]
//                           -     in[IDX(i+1, j)])*dx2i
//                       + ( -     in[IDX(i, j-1)]
//                           + 2.0*in[IDX(i,   j)]
//                           -     in[IDX(i, j+1)])*dy2i;

//     }
//     jm1++;
//     jp1++;
// }

double intemp2[n]={};
double outtemp2[n]={};

for (int j=0; j<n; j++){
intemp2[j]=r[j];
outtemp2[j]=z[j];
}

for (int i=1; i<size; i++){
MPI_Send(intemp2, n, MPI_DOUBLE, i, 513, MPI_COMM_WORLD);
MPI_Send(outtemp2, n, MPI_DOUBLE, i, 514, MPI_COMM_WORLD);
}
}
//At processes other than 0
else{
double intemp2[n]={};
double outtemp2[n]={};
MPI_Recv(intemp2, n, MPI_DOUBLE, 0, 513, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(outtemp2, n, MPI_DOUBLE, 0, 514, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

for (int j=0; j<n; j++){
r[j]=intemp2[j];
z[j]=outtemp2[j];
}     
}


//----------PreCondition end







//Convert Global v, s to Sub vSub, sSub
for (int i=0;i<subGridNum;i++){
rSub[i] = r[subIDXglobal(i, rank)];
zSub[i] = z[subIDXglobal(i, rank)];
} 
//beta = cblas_ddot(n, r, 1, z, 1) / beta; //beta= r^T * z / beta
dot_product = 0.0;
#pragma omp parallel for  reduction(+:dot_product)
for (int i = 0; i < n; ++i) {
dot_product += r[i] * z[i];
}
beta = dot_product / beta;

//cblas_dcopy(n, z, 1, t, 1); //t=z
#pragma omp parallel for 
for (int i = 0; i < n; ++i) {
t[i] = z[i];
}


//cblas_daxpy(n, beta, p, 1, t, 1);//t=beta*p +t  
#pragma omp parallel for 
for (int i = 0; i < n; ++i) {
t[i] += beta * p[i];
}


//cblas_dcopy(n, t, 1, p, 1);//p=t
#pragma omp parallel for 
for (int i = 0; i < n; ++i) {
p[i] = t[i];
}




} while (k < 5000); // Set a maximum number of iterations

//cout<<"CHECK POINT 7 AT PROCESS "<< mygrid_rank<<endl;


if (k == 5000) {
cout << "FAILED TO CONVERGE" << endl;
exit(-1);
}

// //Probe 
// if (rank==0){
//     cout<< "x=s:Streamfuction VALUE = ";
//     for (int i=0; i<n; i++){
//         cout<< x[i] << ", ";
//     }
//     cout<<endl;
// }

//cout << "Converged in " << k << " iterations. eps = " << eps << endl;
}





void SolverCG::ApplyOperator(double* in, double* out) { //in: Streamfunction, out: vorticity

int n = Nx*Ny;

// Assume ordered with y-direction fastest (column-by-column)
double dx2i = 1.0/dx/dx;
double dy2i = 1.0/dy/dy;
int jm1 = 0, jp1 = 2;


// // //num of process per each side
// // sideSubCoreNum = sqrt(size);

// // //Note Example 12.8
// // int sizes[2] = {sideSubCoreNum, sideSubCoreNum};
// // int periods[2] = {0, 0};
// // int reorder = 1;
// // MPI_Cart_create(MPI_COMM_WORLD, 2, sizes, periods, reorder, &mygrid);

// // MPI_Comm_rank(mygrid, &mygrid_rank);
// // MPI_Cart_coords(mygrid, mygrid_rank, 2, coords);

double intemp[n]={};
double outtemp[n]={};
if (rank==0){
for (int j=0; j<n; j++){
intemp[j]=in[j];
outtemp[j]=out[j];
}
for (int i=1; i<size; i++){
MPI_Send(intemp, n, MPI_DOUBLE, i, 33, MPI_COMM_WORLD);
MPI_Send(outtemp, n, MPI_DOUBLE, i, 34, MPI_COMM_WORLD);
}
//cout<<"PROCESS 0 SENT 1"<<endl;
}
else if (rank != 0){
//cout<<"PROCESS "<<mygrid_rank<<"Begines RECEIVED"<<endl;
MPI_Recv(intemp, n, MPI_DOUBLE, 0, 33, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(outtemp, n, MPI_DOUBLE, 0, 34, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


for (int j=0; j<n; j++){
in[j]=intemp[j];
out[j]=outtemp[j];
}     
//cout<<"PROCESS "<<mygrid_rank<<"RECEIVED"<<endl;
}


// //Probe for intemp
// if (rank==0){
//     cout<< "in VALUE at rank 2 = ";
//     for (int i=0; i<n; i++){
//         cout<< in[i] << ", ";
//     }
//     cout<<endl;
// }
// //Probe for outtemp
// if (rank==0){
//     cout<< "out VALUE = ";
//     for (int i=0; i<n; i++){
//         cout<< out[i] << ", ";
//     }
//     cout<<endl;
// }


//Initialize v and s for Sub Cores
double inSub[subGridNum] = {};
double outSub[subGridNum] = {};


//Convert Global v, s to Sub vSub, sSub
#pragma omp parallel for        
for (int i=0;i<subGridNum;i++){
inSub[i] = in[subIDXglobal(i, rank)];
outSub[i] = out[subIDXglobal(i, rank)];
}   


// //Probe
// if (rank==7){
//     cout<< "inSub VALUE = ";
//     for (int i=0; i<subGridNum; i++){
//         cout<< inSub[i] << ", ";
//     }
//     cout<<endl;
//     cout<< "outSub VALUE = ";
//     for (int i=0; i<subGridNum; i++){
//         cout<< outSub[i] << ", ";
//     }
//     cout<<endl;
// }


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


//Communicate with Neighbour Sub Processes
//if sub not at any boundaries
if (coords[0]!=0 && coords[0]!=sideSubCoreNum-1 && coords[1]!=0 && coords[1]!=sideSubCoreNum-1){
for (int i=0; i<subSideNumX; i++){
sSendUp[i] = inSub[i+(subSideNumY-1)*subSideNumX];
sSendDown[i] = inSub[i];
vSendUp[i] = outSub[i+(subSideNumY-1)*subSideNumX];
vSendDown[i] = outSub[i];
}
for (int i=0; i<subSideNumY; i++){
sSendLeft[i] = inSub[i*subSideNumX];
sSendRight[i] = inSub[subSideNumX-1+i*subSideNumX];
vSendLeft[i] = outSub[i*subSideNumX];
vSendRight[i] = outSub[subSideNumX-1+i*subSideNumX];
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
sSendUp[i] = inSub[i+(subSideNumY-1)*subSideNumX];
vSendUp[i] = outSub[i+(subSideNumY-1)*subSideNumX];
}
for (int i=0; i<subSideNumY; i++){
sSendRight[i] = inSub[subSideNumX-1+i*subSideNumX];
vSendRight[i] = outSub[subSideNumX-1+i*subSideNumX];
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
sSendUp[i] = inSub[i+(subSideNumY-1)*subSideNumX];
vSendUp[i] = outSub[i+(subSideNumY-1)*subSideNumX];
}
for (int i=0; i<subSideNumY; i++){
sSendLeft[i] = inSub[i*subSideNumX];
vSendLeft[i] = outSub[i*subSideNumX];
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
sSendDown[i] = inSub[i];
vSendDown[i] = outSub[i];
}
for (int i=0; i<subSideNumY; i++){
sSendRight[i] = inSub[subSideNumX-1+i*subSideNumX];
vSendRight[i] = outSub[subSideNumX-1+i*subSideNumX];
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
sSendDown[i] = inSub[i];
vSendDown[i] = outSub[i];
}
for (int i=0; i<subSideNumY; i++){
sSendLeft[i] = inSub[i*subSideNumX];
vSendLeft[i] = outSub[i*subSideNumX];
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
sSendDown[i] = inSub[i];
vSendDown[i] = outSub[i];
}
for (int i=0; i<subSideNumY; i++){
sSendLeft[i] = inSub[i*subSideNumX];
sSendRight[i] = inSub[subSideNumX-1+i*subSideNumX];
vSendLeft[i] = outSub[i*subSideNumX];
vSendRight[i] = outSub[subSideNumX-1+i*subSideNumX];
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
sSendUp[i] = inSub[i+(subSideNumY-1)*subSideNumX];
vSendUp[i] = outSub[i+(subSideNumY-1)*subSideNumX];
}
for (int i=0; i<subSideNumY; i++){
sSendLeft[i] = inSub[i*subSideNumX];
sSendRight[i] = inSub[subSideNumX-1+i*subSideNumX];
vSendLeft[i] = outSub[i*subSideNumX];
vSendRight[i] = outSub[subSideNumX-1+i*subSideNumX];
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
sSendUp[i] = inSub[i+(subSideNumY-1)*subSideNumX];
sSendDown[i] = inSub[i];
vSendUp[i] = outSub[i+(subSideNumY-1)*subSideNumX];
vSendDown[i] = outSub[i];
}
for (int i=0; i<subSideNumY; i++){
sSendLeft[i] = inSub[i*subSideNumX];
vSendLeft[i] = outSub[i*subSideNumX];
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
sSendUp[i] = inSub[i+(subSideNumY-1)*subSideNumX];
sSendDown[i] = inSub[i];
vSendUp[i] = outSub[i+(subSideNumY-1)*subSideNumX];
vSendDown[i] = outSub[i];
}
for (int i=0; i<subSideNumY; i++){
sSendRight[i] = inSub[subSideNumX-1+i*subSideNumX];
vSendRight[i] = outSub[subSideNumX-1+i*subSideNumX];
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

//cout<<"CHECK POINT 6 PROCESS = "<<rank<<endl;

double inSubtemp3[subGridNum] = {};
double outSubtemp3[subGridNum] = {};

for (int i=0; i<subGridNum;i++){
inSubtemp3[i]=inSub[i];
outSubtemp3[i]=outSub[i];
}

//cout<<"CHECK POINT 7 PROCESS = "<<rank<<endl;

double siPlus = 0;
double siMinus = 0;
double sjPlus = 0;
double sjMinus = 0;
double viPlus = 0;
double viMinus = 0;
double vjPlus = 0;
double vjMinus = 0;

int nthreads, threadid;
#pragma omp parallel for  default(shared) private(siPlus,siMinus,sjPlus,sjMinus,viPlus,viMinus,vjPlus,vjMinus) 
//Handout Algorithm (12)
for (int j = jBegin; j < jEnd; ++j) 
{
for (int i = iBegin; i < iEnd; ++i) {


siPlus = inSubtemp3[subIDX(i+1,j)];
siMinus = inSubtemp3[subIDX(i-1,j)];
sjPlus = inSubtemp3[subIDX(i,j+1)];
sjMinus = inSubtemp3[subIDX(i,j-1)];
viPlus = outSubtemp3[subIDX(i+1,j)];
viMinus = outSubtemp3[subIDX(i-1,j)];
vjPlus = outSubtemp3[subIDX(i,j+1)];
vjMinus = outSubtemp3[subIDX(i,j-1)];


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


outSub[subIDX(i,j)] = ( - siMinus + 2.0*inSub[subIDX(i,j)] - siPlus)*dx2i
+ ( - sjMinus + 2.0*inSub[subIDX(i, j)] - sjPlus)*dy2i;

// //Probe
// if (mygrid_rank==0&&i==1&&j==2){
//     cout<<"result = "<<inSub[subIDX(i,j)]<<endl;
//     cout<<", subIDX = "<< subIDX(i,j)<<", siMinus="<<siMinus<<", siPlus = "<<siPlus<<", sjMius = "<<sjMinus<<", sjPlus = "<<sjPlus<<", p spot = "<< outSub[subIDX(i,j)]<<endl;
// }

// threadid =  omp_get_thread_num();
// cout<<"threadid = "<<threadid<<endl;
// nthreads = omp_get_num_threads();
// cout<<"nthreads = "<<nthreads<<endl;
}

jm1++;
jp1++;
}


// //Probe
// if (rank==0){
//     cout<< "inSub VALUE = ";
//     for (int i=0; i<subGridNum; i++){
//         cout<< inSub[i] << ", ";
//     }
//     cout<<endl;
//     cout<< "outSub VALUE = ";
//     for (int i=0; i<subGridNum; i++){
//         cout<< outSub[i] << ", ";
//     }
//     cout<<endl;
// }


//Gather Sub v to Global at Processor 0 
//If not at Process 0, send u0Sub, u1Sub to Process 0
if (mygrid_rank != 0){
MPI_Send(inSub, subGridNum, MPI_DOUBLE, 0, 233, MPI_COMM_WORLD);
MPI_Send(outSub, subGridNum, MPI_DOUBLE, 0, 243, MPI_COMM_WORLD);
//cout<<"s SENT FROM "<< mygrid_rank << " VALUE = " << sSub[7]<<endl;
}

//If at process 0
if (mygrid_rank == 0){
//cout<<"PROCESS 0 FOR V AND S BEGINS"<<endl;

//for All Process

for (int i = 0; i < size; i++) {
//Process 0
if(i==0){
#pragma omp parallel for        
for (int j=0; j<subGridNum; j++){
in[subIDXglobal(j,0)] = inSub[j];                   
out[subIDXglobal(j,0)] = outSub[j];     
}
}

//At Other Processes
else if (i != 0){
double intemp1[subGridNumArray[i]] = {};
double outtemp1[subGridNumArray[i]] = {};

MPI_Recv(intemp1, subGridNumArray[i], MPI_DOUBLE, i, 233, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(outtemp1, subGridNumArray[i], MPI_DOUBLE, i, 243, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//cout<<"s RECEIVED FROM "<< i << " VALUE = " << stemp[7]<<endl;

#pragma omp parallel for        
for (int j=0; j<subGridNumArray[i]; j++){
in[subIDXglobal(j,i)] = intemp1[j];
out[subIDXglobal(j,i)] = outtemp1[j];
}                
}
}

}
//Gather Sub v to Global at Processor 0



//Broadcast v s from proecess 0
if (mygrid_rank==0){

//     for (int j = 1; j < Ny - 1; ++j) {
//     for (int i = 1; i < Nx - 1; ++i) {
//         out[IDX(i,j)] = ( -     in[IDX(i-1, j)]
//                           + 2.0*in[IDX(i,   j)]
//                           -     in[IDX(i+1, j)])*dx2i
//                       + ( -     in[IDX(i, j-1)]
//                           + 2.0*in[IDX(i,   j)]
//                           -     in[IDX(i, j+1)])*dy2i;

//     }
//     jm1++;
//     jp1++;
// }

double intemp2[n]={};
double outtemp2[n]={};

for (int j=0; j<n; j++){
intemp2[j]=in[j];
outtemp2[j]=out[j];
}

for (int i=1; i<size; i++){
MPI_Send(intemp2, n, MPI_DOUBLE, i, 513, MPI_COMM_WORLD);
MPI_Send(outtemp2, n, MPI_DOUBLE, i, 514, MPI_COMM_WORLD);
}
}
//At processes other than 0
else{
double intemp2[n]={};
double outtemp2[n]={};
MPI_Recv(intemp2, n, MPI_DOUBLE, 0, 513, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(outtemp2, n, MPI_DOUBLE, 0, 514, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

for (int j=0; j<n; j++){
in[j]=intemp2[j];
out[j]=outtemp2[j];
}     
}

// //Probe for in (CHECKED)
// if (rank==2){
//     cout<< "in VALUE at rank 0 = ";
//     for (int i=0; i<n; i++){
//         cout<< in[i] << ", ";
//     }
//     cout<<endl;
// }
// //Probe for out (CHECKED)
// if (rank==2){
//     cout<< "out VALUE = ";
//     for (int i=0; i<n; i++){
//         cout<< out[i] << ", ";
//     }
//     cout<<endl;
// }
}



/* void SolverCG::ApplyOperator(double* in, double* out) {
// Assume ordered with y-direction fastest (column-by-column)
double dx2i = 1.0/dx/dx;
double dy2i = 1.0/dy/dy;
int jm1 = 0, jp1 = 2;
for (int j = 1; j < Ny - 1; ++j) {
for (int i = 1; i < Nx - 1; ++i) {
out[IDX(i,j)] = ( -     in[IDX(i-1, j)]
+ 2.0*in[IDX(i,   j)]
-     in[IDX(i+1, j)])*dx2i
+ ( -     in[IDX(i, j-1)]
+ 2.0*in[IDX(i,   j)]
-     in[IDX(i, j+1)])*dy2i;

}
jm1++;
jp1++;
}
}  */


/* void SolverCG::Precondition(double* in, double* out) {

int n = Nx*Ny;


int i, j;
double dx2i = 1.0/dx/dx;
double dy2i = 1.0/dy/dy;
double factor = 2.0*(dx2i + dy2i);


//
double intemp[n]={};
double outtemp[n]={};
if (rank==0){
for (int j=0; j<n; j++){
intemp[j]=in[j];
outtemp[j]=out[j];
}
for (int i=1; i<size; i++){
MPI_Send(intemp, n, MPI_DOUBLE, i, 33, MPI_COMM_WORLD);
MPI_Send(outtemp, n, MPI_DOUBLE, i, 34, MPI_COMM_WORLD);
}
//cout<<"PROCESS 0 SENT 1"<<endl;
}
else if (rank != 0){
//cout<<"PROCESS "<<mygrid_rank<<"Begines RECEIVED"<<endl;
MPI_Recv(intemp, n, MPI_DOUBLE, 0, 33, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(outtemp, n, MPI_DOUBLE, 0, 34, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


for (int j=0; j<n; j++){
in[j]=intemp[j];
out[j]=outtemp[j];
}     
//cout<<"PROCESS "<<mygrid_rank<<"RECEIVED"<<endl;
}


// //Probe for intemp
// if (rank==0){
//     cout<< "in VALUE at rank 2 = ";
//     for (int i=0; i<n; i++){
//         cout<< in[i] << ", ";
//     }
//     cout<<endl;
// }
// //Probe for outtemp
// if (rank==0){
//     cout<< "out VALUE = ";
//     for (int i=0; i<n; i++){
//         cout<< out[i] << ", ";
//     }
//     cout<<endl;
// }


//Initialize v and s for Sub Cores
double inSub[subGridNum] = {};
double outSub[subGridNum] = {};

//Convert Global v, s to Sub vSub, sSub
for (int i=0;i<subGridNum;i++){
inSub[i] = in[subIDXglobal(i, rank)];
outSub[i] = out[subIDXglobal(i, rank)];
}   


// //Probe
// if (rank==7){
//     cout<< "inSub VALUE = ";
//     for (int i=0; i<subGridNum; i++){
//         cout<< inSub[i] << ", ";
//     }
//     cout<<endl;
//     cout<< "outSub VALUE = ";
//     for (int i=0; i<subGridNum; i++){
//         cout<< outSub[i] << ", ";
//     }
//     cout<<endl;
// }//test sub position when alogrithm nelegecting boundary
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

// Boundaries, let out = in 
for (i = 0; i < subSideNumX; ++i) {
outSub[subIDX(i, 0)] = inSub[subIDX(i,0)];
//            outSub[subIDX(i, subSideNumY-1)] = inSub[subIDX(i, subSideNumY-1)];
}
for (j = 0; j < subSideNumY; ++j) {
outSub[subIDX(0, j)] = inSub[subIDX(0, j)];
//            outSub[subIDX(subSideNumX - 1, j)] = inSub[subIDX(subSideNumX - 1, j)];
}
}
//if sub at bottom right corner
else if (coords[0]==0 && coords[1]==sideSubCoreNum-1){
iBegin = 0;
iEnd = subSideNumX-1;
jBegin = 1;
jEnd = subSideNumY; 

// Boundaries, let out = in 
for (i = 0; i < subSideNumX; ++i) {
outSub[subIDX(i, 0)] = inSub[subIDX(i,0)];
//            outSub[subIDX(i, subSideNumY-1)] = inSub[subIDX(i, subSideNumY-1)];
}
for (j = 0; j < subSideNumY; ++j) {
//            outSub[subIDX(0, j)] = inSub[subIDX(0, j)];
outSub[subIDX(subSideNumX - 1, j)] = inSub[subIDX(subSideNumX - 1, j)];
}
}
//if sub at top left corner
else if (coords[0]==sideSubCoreNum-1 && coords[1]==0){
iBegin = 1;
iEnd = subSideNumX;
jBegin = 0;
jEnd = subSideNumY-1; 

// Boundaries, let out = in 
for (i = 0; i < subSideNumX; ++i) {
//            outSub[subIDX(i, 0)] = inSub[subIDX(i,0)];
outSub[subIDX(i, subSideNumY-1)] = inSub[subIDX(i, subSideNumY-1)];
}
for (j = 0; j < subSideNumY; ++j) {
outSub[subIDX(0, j)] = inSub[subIDX(0, j)];
//            outSub[subIDX(subSideNumX - 1, j)] = inSub[subIDX(subSideNumX - 1, j)];
}
}
//if sub at top right corner
else if (coords[0]==sideSubCoreNum-1 && coords[1]==sideSubCoreNum-1){
iBegin = 0;
iEnd = subSideNumX-1;
jBegin = 0;
jEnd = subSideNumY-1; 

// Boundaries, let out = in 
for (i = 0; i < subSideNumX; ++i) {
//            outSub[subIDX(i, 0)] = inSub[subIDX(i,0)];
outSub[subIDX(i, subSideNumY-1)] = inSub[subIDX(i, subSideNumY-1)];
}
for (j = 0; j < subSideNumY; ++j) {
//            outSub[subIDX(0, j)] = inSub[subIDX(0, j)];
outSub[subIDX(subSideNumX - 1, j)] = inSub[subIDX(subSideNumX - 1, j)];
}
}
//if sub at top edge
else if (coords[0]==sideSubCoreNum-1 && coords[1]!=sideSubCoreNum-1 && coords[1]!=0){
iBegin = 0;
iEnd = subSideNumX;
jBegin = 0;
jEnd = subSideNumY-1; 

// Boundaries, let out = in 
for (i = 0; i < subSideNumX; ++i) {
//            outSub[subIDX(i, 0)] = inSub[subIDX(i,0)];
outSub[subIDX(i, subSideNumY-1)] = inSub[subIDX(i, subSideNumY-1)];
}
for (j = 0; j < subSideNumY; ++j) {
//            outSub[subIDX(0, j)] = inSub[subIDX(0, j)];
//            outSub[subIDX(subSideNumX - 1, j)] = inSub[subIDX(subSideNumX - 1, j)];
}
}
//if sub at bottom edge
else if (coords[0]==0 && coords[1]!=sideSubCoreNum-1 && coords[1]!=0){
iBegin = 0;
iEnd = subSideNumX;
jBegin = 1;
jEnd = subSideNumY; 

// Boundaries, let out = in 
for (i = 0; i < subSideNumX; ++i) {
outSub[subIDX(i, 0)] = inSub[subIDX(i,0)];
//            outSub[subIDX(i, subSideNumY-1)] = inSub[subIDX(i, subSideNumY-1)];
}
for (j = 0; j < subSideNumY; ++j) {
//            outSub[subIDX(0, j)] = inSub[subIDX(0, j)];
//            outSub[subIDX(subSideNumX - 1, j)] = inSub[subIDX(subSideNumX - 1, j)];
}
}
//if sub at right edge
else if (coords[1]==sideSubCoreNum-1 && coords[0]!=sideSubCoreNum-1 && coords[0]!=0){
iBegin = 0;
iEnd = subSideNumX-1;
jBegin = 0;
jEnd = subSideNumY; 

// Boundaries, let out = in 
for (i = 0; i < subSideNumX; ++i) {
//            outSub[subIDX(i, 0)] = inSub[subIDX(i,0)];
//            outSub[subIDX(i, subSideNumY-1)] = inSub[subIDX(i, subSideNumY-1)];
}
for (j = 0; j < subSideNumY; ++j) {
//            outSub[subIDX(0, j)] = inSub[subIDX(0, j)];
outSub[subIDX(subSideNumX - 1, j)] = inSub[subIDX(subSideNumX - 1, j)];
}
}
//if sub at left edge
else if (coords[1]==0 && coords[0]!=sideSubCoreNum-1 && coords[0]!=0){
iBegin = 1;
iEnd = subSideNumX;
jBegin = 0;
jEnd = subSideNumY; 
// Boundaries, let out = in 
for (i = 0; i < subSideNumX; ++i) {
//            outSub[subIDX(i, 0)] = inSub[subIDX(i,0)];
//            outSub[subIDX(i, subSideNumY-1)] = inSub[subIDX(i, subSideNumY-1)];
}
for (j = 0; j < subSideNumY; ++j) {
outSub[subIDX(0, j)] = inSub[subIDX(0, j)];
//            outSub[subIDX(subSideNumX - 1, j)] = inSub[subIDX(subSideNumX - 1, j)];
}
}


// int nthreads, threadid;
#pragma omp parallel for 
//For interior, let out = in / factor
for (int j = jBegin; j < jEnd; ++j) 
{
for (int i = iBegin; i < iEnd; ++i) {
outSub[subIDX(i,j)] = inSub[subIDX(i,j)]/factor;
// threadid =  omp_get_thread_num();
// cout<<"threadid = "<<threadid<<endl;
// nthreads = omp_get_num_threads();
// cout<<"nthreads = "<<nthreads<<endl;
}
}


//Gather Sub v to Global at Processor 0 
//If not at Process 0, send u0Sub, u1Sub to Process 0
if (mygrid_rank != 0){
MPI_Send(inSub, subGridNum, MPI_DOUBLE, 0, 233, MPI_COMM_WORLD);
MPI_Send(outSub, subGridNum, MPI_DOUBLE, 0, 243, MPI_COMM_WORLD);
//cout<<"s SENT FROM "<< mygrid_rank << " VALUE = " << sSub[7]<<endl;
}

//If at process 0
if (mygrid_rank == 0){
//cout<<"PROCESS 0 FOR V AND S BEGINS"<<endl;

//for All Process
for (int i = 0; i < size; i++) {

//Process 0
if(i==0){
#pragma omp parallel for        
for (int j=0; j<subGridNum; j++){
in[subIDXglobal(j,0)] = inSub[j];                   
out[subIDXglobal(j,0)] = outSub[j];     
}
}

//At Other Processes
else if (i != 0){
double intemp1[subGridNumArray[i]] = {};
double outtemp1[subGridNumArray[i]] = {};

MPI_Recv(intemp1, subGridNumArray[i], MPI_DOUBLE, i, 233, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(outtemp1, subGridNumArray[i], MPI_DOUBLE, i, 243, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//cout<<"s RECEIVED FROM "<< i << " VALUE = " << stemp[7]<<endl;

#pragma omp parallel for        
for (int j=0; j<subGridNumArray[i]; j++){
in[subIDXglobal(j,i)] = intemp1[j];
out[subIDXglobal(j,i)] = outtemp1[j];
}                
}
}

}
//Gather Sub v to Global at Processor 0



//Broadcast v s from proecess 0
if (mygrid_rank==0){

//     for (int j = 1; j < Ny - 1; ++j) {
//     for (int i = 1; i < Nx - 1; ++i) {
//         out[IDX(i,j)] = ( -     in[IDX(i-1, j)]
//                           + 2.0*in[IDX(i,   j)]
//                           -     in[IDX(i+1, j)])*dx2i
//                       + ( -     in[IDX(i, j-1)]
//                           + 2.0*in[IDX(i,   j)]
//                           -     in[IDX(i, j+1)])*dy2i;

//     }
//     jm1++;
//     jp1++;
// }

double intemp2[n]={};
double outtemp2[n]={};

for (int j=0; j<n; j++){
intemp2[j]=in[j];
outtemp2[j]=out[j];
}

for (int i=1; i<size; i++){
MPI_Send(intemp2, n, MPI_DOUBLE, i, 513, MPI_COMM_WORLD);
MPI_Send(outtemp2, n, MPI_DOUBLE, i, 514, MPI_COMM_WORLD);
}
}
//At processes other than 0
else{
double intemp2[n]={};
double outtemp2[n]={};
MPI_Recv(intemp2, n, MPI_DOUBLE, 0, 513, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(outtemp2, n, MPI_DOUBLE, 0, 514, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

for (int j=0; j<n; j++){
in[j]=intemp2[j];
out[j]=outtemp2[j];
}     
}


} */

/* 
void SolverCG::Precondition(double* in, double* out) {
int i, j;
double dx2i = 1.0/dx/dx;
double dy2i = 1.0/dy/dy;
double factor = 2.0*(dx2i + dy2i);
for (i = 1; i < Nx - 1; ++i) {
for (j = 1; j < Ny - 1; ++j) {
out[IDX(i,j)] = in[IDX(i,j)]/factor;
}
}
// Boundaries
for (i = 0; i < Nx; ++i) {
out[IDX(i, 0)] = in[IDX(i,0)];
out[IDX(i, Ny-1)] = in[IDX(i, Ny-1)];
}

for (j = 0; j < Ny; ++j) {
out[IDX(0, j)] = in[IDX(0, j)];
out[IDX(Nx - 1, j)] = in[IDX(Nx - 1, j)];
}
} */



// //let vorticity at Boundary to be 0
// void SolverCG::ImposeBC(double* inout) {

//         int n = Nx*Ny;


//     if (rank==0){
//         // int nthreads, threadid;  

//         #pragma omp parallel for 
//             // Boundaries
//         for (int i = 0; i < Nx; ++i) {
//             inout[IDX(i, 0)] = 0.0;
//             inout[IDX(i, Ny-1)] = 0.0;
//             // threadid=omp_get_thread_num();
//             // cout<<"threadid = "<<threadid<<endl;
//         }

//         #pragma omp parallel for 
//         for (int j = 0; j < Ny; ++j) {
//             inout[IDX(0, j)] = 0.0;
//             inout[IDX(Nx - 1, j)] = 0.0;
//         }
//     }
// }
