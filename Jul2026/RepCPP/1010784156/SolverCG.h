#pragma once

#include <mpi.h>
#include <cmath>

class SolverCG
{
public:
    SolverCG(int pNx, int pNy, double pdx, double pdy);
    ~SolverCG();

    void Solve(double* b, double* x); //b is vorticity, x is streamfunction
    //
    int rank = 0;
    int size = 0;
private:
    double dx;
    double dy;
    int Nx;
    int Ny;
    double* r;
    double* p;
    double* z;
    double* t;

    int Npts = 81;

    void ApplyOperator(double* p, double* t);
    void Precondition(double* p, double* t);
    void ImposeBC(double* p);
    void probe(int K, int RANK, int k, int rank, double* x, double* t, double* p, int n, double* z, double alpha, double beta) ;


    int subIDXglobal(int iSub, int rank);

    
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


};

