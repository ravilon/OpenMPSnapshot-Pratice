#if !defined(AFX_SERIALIZARE_BARIERA_H__1022744A_0EDE_4F4E_A0B7_28AC6B28B018__INCLUDED_)
#define AFX_SERIALIZARE_BARIERA_H__1022744A_0EDE_4F4E_A0B7_28AC6B28B018__INCLUDED_
#include "bariera.h"
#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
// serializare_bariera.h : header file
//



/////////////////////////////////////////////////////////////////////////////
// serializare_bariera thread

class serializare_bariera : public CWinThread
{
	DECLARE_DYNCREATE(serializare_bariera)
protected:
	serializare_bariera();           // protected constructor used by dynamic creation

// Attributes
public:

// Operations
public:

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(serializare_bariera)
	public:
	virtual BOOL InitInstance();
	serializare_bariera(int nr,CSemaphore *mtx,int *indexare,bariera *bar);
	virtual int ExitInstance();
	int Run();
	//}}AFX_VIRTUAL

// Implementation
protected:
	virtual ~serializare_bariera();

	// Generated message map functions
	//{{AFX_MSG(serializare_bariera)
		// NOTE - the ClassWizard will add and remove member functions here.
	//}}AFX_MSG
private:
	CSemaphore* sem;
	int threads;
	int index;
	bariera *barrier;
	DECLARE_MESSAGE_MAP()
};

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_SERIALIZARE_BARIERA_H__1022744A_0EDE_4F4E_A0B7_28AC6B28B018__INCLUDED_)
