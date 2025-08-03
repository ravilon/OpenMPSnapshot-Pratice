// serializare.h: interface for the serializare class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_SERIALIZARE_H__2078806F_DC12_4508_8CA2_2D83FE0DAEAD__INCLUDED_)
#define AFX_SERIALIZARE_H__2078806F_DC12_4508_8CA2_2D83FE0DAEAD__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#define N 4
#include <afxmt.h>

class serializare : public CWinThread
{
public:
serializare();
virtual ~serializare();
private:
int ordine[N];
CSemaphore sem[N];
CMutex	mutex[N];
HANDLE hthreads[N];
CEvent event[N];
CCriticalSection section[N];
};

#endif // !defined(AFX_SERIALIZARE_H__2078806F_DC12_4508_8CA2_2D83FE0DAEAD__INCLUDED_)
