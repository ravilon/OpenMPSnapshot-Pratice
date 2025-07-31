#if !defined(AFX_THREAD_PROD_H__28096C5E_AFC4_41A1_B723_713046904447__INCLUDED_)
#define AFX_THREAD_PROD_H__28096C5E_AFC4_41A1_B723_713046904447__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
// thread_prod.h : header file
//



/////////////////////////////////////////////////////////////////////////////
// thread_prod thread

class thread_prod : public CWinThread
{
DECLARE_DYNCREATE(thread_prod)
protected:
thread_prod();           // protected constructor used by dynamic creation

// Attributes
public:

// Operations
public:
int Run();
thread_prod(long dimensiune,int proc,double **mat0,double **mat1,double **mat2,int *indexare);
// Overrides
// ClassWizard generated virtual function overrides
//{{AFX_VIRTUAL(thread_prod)
public:
virtual BOOL InitInstance();
virtual int ExitInstance();
//}}AFX_VIRTUAL

// Implementation
protected:
virtual ~thread_prod();

// Generated message map functions
//{{AFX_MSG(thread_prod)
// NOTE - the ClassWizard will add and remove member functions here.
//}}AFX_MSG

DECLARE_MESSAGE_MAP()
private:
long dim;
int threads;
double **mata;
double **matb;
double **matc;
int index;
};

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_THREAD_PROD_H__28096C5E_AFC4_41A1_B723_713046904447__INCLUDED_)
