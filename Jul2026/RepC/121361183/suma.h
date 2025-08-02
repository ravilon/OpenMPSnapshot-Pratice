#if !defined(AFX_SUMA_H__FA7A27C5_BBB7_4838_9D10_EB3C3702D010__INCLUDED_)
#define AFX_SUMA_H__FA7A27C5_BBB7_4838_9D10_EB3C3702D010__INCLUDED_
#include <afxmt.h>
#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
// suma.h : header file
//



/////////////////////////////////////////////////////////////////////////////
// suma thread

class suma : public CWinThread
{
	DECLARE_DYNCREATE(suma)
protected:
	suma();           // protected constructor used by dynamic creation

// Attributes
public:

// Operations
public:
	int Run();

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(suma)
	public:
	suma(long dimensiune,int nrthreads,int *vectori,CMutex *mutexuri,long *paralel,int *poz);
	virtual BOOL InitInstance();
	virtual int ExitInstance();
	//}}AFX_VIRTUAL

// Implementation
protected:
	virtual ~suma();

	// Generated message map functions
	//{{AFX_MSG(suma)
		// NOTE - the ClassWizard will add and remove member functions here.
	//}}AFX_MSG

	DECLARE_MESSAGE_MAP()
private:
	long  *rezultat;
	long dim;
	int threads;
	int *vector;
	CMutex *mutex;
	int index;
};

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_SUMA_H__FA7A27C5_BBB7_4838_9D10_EB3C3702D010__INCLUDED_)
