#if !defined(AFX_SERIALIZARE_CRITICALS_H__C57703C9_9F4B_4537_9DF2_766EA9C58872__INCLUDED_)
#define AFX_SERIALIZARE_CRITICALS_H__C57703C9_9F4B_4537_9DF2_766EA9C58872__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
// serializare_criticals.h : header file
//

#include <afxmt.h>

/////////////////////////////////////////////////////////////////////////////
// serializare_criticals thread

class serializare_criticals : public CWinThread
{
	DECLARE_DYNCREATE(serializare_criticals)
protected:
	serializare_criticals();           // protected constructor used by dynamic creation

// Attributes
public:

// Operations
public:
	int Run();

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(serializare_criticals)
	public:
	serializare_criticals(int nr,CCriticalSection *crit,int *indexare);
	virtual BOOL InitInstance();
	virtual int ExitInstance();
	//}}AFX_VIRTUAL

// Implementation
protected:
	virtual ~serializare_criticals();

	// Generated message map functions
	//{{AFX_MSG(serializare_criticals)
		// NOTE - the ClassWizard will add and remove member functions here.
	//}}AFX_MSG

	DECLARE_MESSAGE_MAP()
private:
	int index;
	int threads;
	CCriticalSection *critical;
};

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_SERIALIZARE_CRITICALS_H__C57703C9_9F4B_4537_9DF2_766EA9C58872__INCLUDED_)
