#if !defined(AFX_THREAD_CPY_H__7CA72181_E97E_40DE_9E7E_8A141A6FC68E__INCLUDED_)
#define AFX_THREAD_CPY_H__7CA72181_E97E_40DE_9E7E_8A141A6FC68E__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
// thread_cpy.h : header file
//



/////////////////////////////////////////////////////////////////////////////
// thread_cpy thread

class thread_cpy : public CWinThread
{
DECLARE_DYNCREATE(thread_cpy)
protected:
thread_cpy();           // protected constructor used by dynamic creation

// Attributes
public:
private:
long dimensiune;
int thread;
double *freev;
double *freev1;
double **matr1;
double **matr;
long pos;

// Operations
public:
int Run(void);

// Overrides
// ClassWizard generated virtual function overrides
//{{AFX_VIRTUAL(thread_cpy)
public:
thread_cpy(long dim,int threads,double *y,double *y1,double **mat,double **mat1,long *index);
virtual BOOL InitInstance();
virtual int ExitInstance();
//}}AFX_VIRTUAL

// Implementation
protected:
virtual ~thread_cpy();

// Generated message map functions
//{{AFX_MSG(thread_cpy)
// NOTE - the ClassWizard will add and remove member functions here.
//}}AFX_MSG

DECLARE_MESSAGE_MAP();
};

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_THREAD_CPY_H__7CA72181_E97E_40DE_9E7E_8A141A6FC68E__INCLUDED_)
