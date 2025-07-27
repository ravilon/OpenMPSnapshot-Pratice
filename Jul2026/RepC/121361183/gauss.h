// gauss.h: interface for the gauss class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_GAUSS_H__953CCD66_737A_4480_B76C_95AC40B14FFA__INCLUDED_)
#define AFX_GAUSS_H__953CCD66_737A_4480_B76C_95AC40B14FFA__INCLUDED_
#include <afxmt.h>
#include "bariera.h"
//#include "solveGE.h"
#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

class gauss  
{
public:
	static UINT calcul_tgauss(LPVOID pParam);
	gauss();
	gauss(long dim,int thread,double **mat,double *x,double *y);
	virtual ~gauss();
private:
	long *data;
	CCriticalSection *crit;
	int *counters;
	long i;
	HANDLE *hthreads;
	bariera barrier;
//	solveGE *threads;
	struct datag
	{
		double **mat;
		double *y;
		CCriticalSection *crit;
		int who;
		long N;
		int P;
		int *counters;
		bariera *barrier;
	};
};

#endif // !defined(AFX_GAUSS_H__953CCD66_737A_4480_B76C_95AC40B14FFA__INCLUDED_)
