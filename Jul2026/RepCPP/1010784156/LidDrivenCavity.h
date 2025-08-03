#pragma once

#include <cmath>
#include <mpi.h>
#include<array>

#include <string>
using namespace std;


class SolverCG;

class LidDrivenCavity
{
public:
    LidDrivenCavity();
    ~LidDrivenCavity();

    void SetDomainSize(double xlen, double ylen);
    void SetGridSize(int nx, int ny);
    void SetTimeStep(double deltat);
    void SetFinalTime(double finalt);
    void SetReynoldsNumber(double Re);


    //
    void SetSubCore();
    //


    void Initialise();
    void Integrate();
    void WriteSolution(std::string file);
    void PrintConfiguration();
    //
    int subIDXglobal(int iSub, int rank);
    void Advance(double* v, double* s, SolverCG* cg, double dx, double dy);
    //
        //
    int rank = 0;
    int size = 0;

private:
    double* v   = nullptr;
    double* s   = nullptr;
    double* tmp = nullptr;

    //
    int* subGridNumArray = nullptr;
    int* subSideNumXArray = nullptr;
    int* subSideNumYArray = nullptr;





  
    int coords[2];
    int keep[2];

    int sideSubCoreNum = 0;
    int subGridNum = 0;

    int subSideNum = 0;
    int subSideNumX = 0;
    int subSideNumY = 0;
    

    MPI_Comm mygrid;

    MPI_Comm myrowcomm, mycolcomm;

    int mygrid_rank;
    //




    double dt   = 0.01;
    double T    = 1.0;
    double dx;
    double dy;
    int    Nx   = 9;
    int    Ny   = 9;
    int    Npts = 81;
    double Lx   = 1.0;
    double Ly   = 1.0;
    double Re   = 10;
    double U    = 1.0;
    double nu   = 0.1;

    SolverCG* cg = nullptr;

    void CleanUp();
    void UpdateDxDy();


};

