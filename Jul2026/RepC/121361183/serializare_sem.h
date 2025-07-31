#if !defined(AFX_SERIALIZARE_SEM_H__1D897446_F974_4A45_9C86_36E18E745C36__INCLUDED_)
#define AFX_SERIALIZARE_SEM_H__1D897446_F974_4A45_9C86_36E18E745C36__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
// serializare_sem.h : header file
//

#include <afxmt.h>

/////////////////////////////////////////////////////////////////////////////
// serializare_sem thread

class serializare_sem : public CWinThread
{
DECLARE_DYNCREATE(serializare_sem)
protected:
serializare_sem();           // protected constructor used by dynamic creation

// Attributes
public:

// Operations
public:

// Overrides
// ClassWizard generated virtual function overrides
//{{AFX_VIRTUAL(serializare_sem)
public:
virtual BOOL InitInstance();
virtual BOOL InitInstance(int *indexare);
virtual int ExitInstance();
serializare_sem(int nr,CSemaphore *semaphore);
//}}AFX_VIRTUAL
int Run();
// Implementation
protected:
virtual ~serializare_sem();

// Generated message map functions
//{{AFX_MSG(serializare_sem)
// NOTE - the ClassWizard will add and remove member functions here.
//}}AFX_MSG

DECLARE_MESSAGE_MAP()
private:
CSemaphore* sem;
int index;
int threads;
};

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_SERIALIZARE_SEM_H__1D897446_F974_4A45_9C86_36E18E745C36__INCLUDED_)
