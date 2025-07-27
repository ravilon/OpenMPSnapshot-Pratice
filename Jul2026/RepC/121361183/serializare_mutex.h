#if !defined(AFX_SERIALIZARE_MUTEX_H__EA6E98C3_D170_4473_8FE1_1DC806878B05__INCLUDED_)
#define AFX_SERIALIZARE_MUTEX_H__EA6E98C3_D170_4473_8FE1_1DC806878B05__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
// serializare_mutex.h : header file
//

#include <afxmt.h>

/////////////////////////////////////////////////////////////////////////////
// serializare_mutex thread

class serializare_mutex : public CWinThread
{
	DECLARE_DYNCREATE(serializare_mutex)
protected:
	serializare_mutex();           // protected constructor used by dynamic creation

// Attributes
public:

// Operations
public:

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(serializare_mutex)
	public:
	virtual BOOL InitInstance();
	serializare_mutex(int nr,CMutex *mtx,int *indexare);
	virtual int ExitInstance();
	int Run();
	//}}AFX_VIRTUAL

// Implementation
protected:
	virtual ~serializare_mutex();
	// Generated message map functions
	//{{AFX_MSG(serializare_mutex)
		// NOTE - the ClassWizard will add and remove member functions here.
	//}}AFX_MSG

	DECLARE_MESSAGE_MAP()
private:
	CMutex* mutexuri;
	int threads;
	int index;
};

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_SERIALIZARE_MUTEX_H__EA6E98C3_D170_4473_8FE1_1DC806878B05__INCLUDED_)
