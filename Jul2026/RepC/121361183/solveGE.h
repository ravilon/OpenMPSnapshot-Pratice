#if !defined(AFX_SOLVEGE_H__F456FCF5_F5C0_41BD_BDC5_C8ABAA7267DB__INCLUDED_)
#define AFX_SOLVEGE_H__F456FCF5_F5C0_41BD_BDC5_C8ABAA7267DB__INCLUDED_
#include <afxmt.h>
#include "bariera.h"
#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
// solveGE.h : header file
//



/////////////////////////////////////////////////////////////////////////////
// solveGE thread

class solveGE : public CWinThread
{
DECLARE_DYNCREATE(solveGE)
protected:
solveGE();           // protected constructor used by dynamic creation
CCriticalSection *crit;
double **mat;
double *y;
int who;
long N;
int P;
int *counters;
bariera *barrier;
// Attributes
public:

// Operations
public:
int Run(void);

// Overrides
// ClassWizard generated virtual function overrides
//{{AFX_VIRTUAL(solveGE)
public:
solveGE(double **matrix,double *yterm,long dim,int dim_threads,CCriticalSection *critical,int *count,bariera *bar,int whoindex);
virtual BOOL InitInstance();
virtual int ExitInstance();
//}}AFX_VIRTUAL

// Implementation
protected:
virtual ~solveGE();

// Generated message map functions
//{{AFX_MSG(solveGE)
// NOTE - the ClassWizard will add and remove member functions here.
//}}AFX_MSG

DECLARE_MESSAGE_MAP()
private:
long s;				//Nr of row in partition
long nr;			//average nr of row in each partition
long lastrow;		//if this thread have the last line
long proccount;		//variable for eliminate the row=0 and last row
long proccount1;	//variable who represent the processor who is working
long replay;		//variable for continue the elimination where it was stop
long i,j,k,l;		//counters
};

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_SOLVEGE_H__F456FCF5_F5C0_41BD_BDC5_C8ABAA7267DB__INCLUDED_)
