#if !defined(AFX_SERIALIZA_SEM_H__02491FCA_32FD_4976_B1B3_6B66BD8C69DC__INCLUDED_)
#define AFX_SERIALIZA_SEM_H__02491FCA_32FD_4976_B1B3_6B66BD8C69DC__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
// serializa_sem.h : header file
//

#include <afxmt.h>

/////////////////////////////////////////////////////////////////////////////
// serializa_sem thread

class serializa_sem : public CWinThread
{
DECLARE_DYNCREATE(serializa_sem)
protected:
serializa_sem();           // protected constructor used by dynamic creation
serializa

// Attributes
public:

// Operations
public:

// Overrides
// ClassWizard generated virtual function overrides
//{{AFX_VIRTUAL(serializa_sem)
public:
virtual BOOL InitInstance();
virtual int ExitInstance();
//}}AFX_VIRTUAL

// Implementation
protected:
virtual ~serializa_sem();

// Generated message map functions
//{{AFX_MSG(serializa_sem)
// NOTE - the ClassWizard will add and remove member functions here.
//}}AFX_MSG

DECLARE_MESSAGE_MAP()
};

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_SERIALIZA_SEM_H__02491FCA_32FD_4976_B1B3_6B66BD8C69DC__INCLUDED_)
