// bariera.h: interface for the bariera class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_bariera_H__DD0FF4F1_C3B5_4E43_BD8F_124915C96EC6__INCLUDED_)
#define AFX_bariera_H__DD0FF4F1_C3B5_4E43_BD8F_124915C96EC6__INCLUDED_
#include <afxmt.h>
#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

class bariera  
{
public:
	bariera();
	virtual ~bariera();
	int nWaiters;		/* nr of threads in waiting */
	int bCount;			/* threads counter */
	CCriticalSection bMutex;
	CEvent *bCond;
	int barrier_init(int Count);
	int barrier_reset(int Count);
	int barrier_sync(void);
	void barrier_destroy(void);
};

#endif // !defined(AFX_bariera_H__DD0FF4F1_C3B5_4E43_BD8F_124915C96EC6__INCLUDED_)
