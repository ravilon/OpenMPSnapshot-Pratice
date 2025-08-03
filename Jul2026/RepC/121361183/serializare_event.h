#if !defined(AFX_SERIALIZARE_EVENT_H__CA8B1DAC_7830_432E_8E07_C3DFBC1DDDDF__INCLUDED_)
#define AFX_SERIALIZARE_EVENT_H__CA8B1DAC_7830_432E_8E07_C3DFBC1DDDDF__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
// serializare_event.h : header file
//
#include <afxmt.h>


/////////////////////////////////////////////////////////////////////////////
// serializare_event thread

class serializare_event : public CWinThread
{
DECLARE_DYNCREATE(serializare_event)
protected:
serializare_event();           // protected constructor used by dynamic creation

// Attributes
public:

// Operations
public:
int Run();

// Overrides
// ClassWizard generated virtual function overrides
//{{AFX_VIRTUAL(serializare_event)
public:
serializare_event(int nr,CEvent *ev,int *indexare);
virtual BOOL InitInstance();
virtual int ExitInstance();
//}}AFX_VIRTUAL

// Implementation
protected:
virtual ~serializare_event();

// Generated message map functions
//{{AFX_MSG(serializare_event)
// NOTE - the ClassWizard will add and remove member functions here.
//}}AFX_MSG
private:
int index;
CEvent *event;
int threads;
DECLARE_MESSAGE_MAP()
};

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_SERIALIZARE_EVENT_H__CA8B1DAC_7830_432E_8E07_C3DFBC1DDDDF__INCLUDED_)
