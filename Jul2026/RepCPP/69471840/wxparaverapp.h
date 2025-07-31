/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  wxparaver                                *
 *              Paraver Trace Visualization and Analysis Tool                *
 *****************************************************************************
 *     ___     This library is free software; you can redistribute it and/or *
 *    /  __         modify it under the terms of the GNU LGPL as published   *
 *   /  /  _____    by the Free Software Foundation; either version 2.1      *
 *  /  /  /     \   of the License, or (at your option) any later version.   *
 * (  (  ( B S C )                                                           *
 *  \  \  \_____/   This library is distributed in hope that it will be      *
 *   \  \__         useful but WITHOUT ANY WARRANTY; without even the        *
 *    \___          implied warranty of MERCHANTABILITY or FITNESS FOR A     *
 *                  PARTICULAR PURPOSE. See the GNU LGPL for more details.   *
 *                                                                           *
 * You should have received a copy of the GNU Lesser General Public License  *
 * along with this library; if not, write to the Free Software Foundation,   *
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA          *
 * The GNU LEsser General Public License is contained in the file COPYING.   *
 *                                 ---------                                 *
 *   Barcelona Supercomputing Center - Centro Nacional de Supercomputacion   *
\*****************************************************************************/

#pragma once



/*!
 * Includes
 */

////@begin includes
#include "wx/image.h"
#include "paravermain.h"
////@end includes
#include <wx/snglinst.h>
#include "wx/dirctrl.h"
#include "wx/filefn.h"
#include <time.h>
#ifdef TRACING_ENABLED
#include "extrae_user_events.h"
#endif

// find a common place; moved from paravermain.cpp
#ifdef _WIN32
  const char PATH_SEP('\\');
#else
  const char PATH_SEP('/');
#endif
  
/*!
 * Forward declarations
 */

////@begin forward declarations
////@end forward declarations
class stServer;

/*!
 * Control identifiers
 */

////@begin control identifiers
////@end control identifiers

bool launchBrowser( const wxString& htmlFile );

/*!
 * wxparaverApp class declaration
 */

class wxparaverApp: public wxApp
{    
    DECLARE_CLASS( wxparaverApp )
    DECLARE_EVENT_TABLE()

public:
    /// Constructor
    wxparaverApp();

    void Init();

    /// Initialises the application
    virtual bool OnInit();

    virtual int OnRun();

    /// Called on exit
    virtual int OnExit();

    int FilterEvent(wxEvent& event);

#if !defined _MSC_VER && !defined __MINGW32__
    // Signal handling
    static void handler( int signum );
    void presetUserSignals();
#endif

    void ActivateGlobalTiming( wxDialog* whichDialog );
    void DeactivateGlobalTiming();
    
    void ParseCommandLine( wxCmdLineParser& paraverCommandLineParser );

////@begin wxparaverApp event handler declarations

////@end wxparaverApp event handler declarations

////@begin wxparaverApp member function declarations

	TEventType GetEventTypeForCode() const { return eventTypeForCode ; }
	void SetEventTypeForCode(TEventType value) { eventTypeForCode = value ; }

	bool GetGlobalTiming() const { return globalTiming ; }
	void SetGlobalTiming(bool value) { globalTiming = value ; }

	TTime GetGlobalTimingBegin() const { return globalTimingBegin ; }
	void SetGlobalTimingBegin(TTime value) { globalTimingBegin = value ; }

	bool GetGlobalTimingBeginIsSet() const { return globalTimingBeginIsSet ; }
	void SetGlobalTimingBeginIsSet(bool value) { globalTimingBeginIsSet = value ; }

	wxDialog* GetGlobalTimingCallDialog() const { return globalTimingCallDialog ; }
	void SetGlobalTimingCallDialog(wxDialog* value) { globalTimingCallDialog = value ; }

	TTime GetGlobalTimingEnd() const { return globalTimingEnd ; }
	void SetGlobalTimingEnd(TTime value) { globalTimingEnd = value ; }

////@end wxparaverApp member function declarations

    static paraverMain* mainWindow;
    static wxCmdLineEntryDesc argumentsParseSyntax[];
    void ValidateSession( bool setValidate );
    
////@begin wxparaverApp member variables
private:
	TEventType eventTypeForCode;
	bool globalTiming;
	TTime globalTimingBegin;
	bool globalTimingBeginIsSet;
	wxDialog* globalTimingCallDialog;
	TTime globalTimingEnd;
////@end wxparaverApp member variables
    bool invalidateNoConnect;

    wxLocale m_locale;
    
    wxSingleInstanceChecker *m_checker;
    
    stServer *m_server;
    
    void PrintVersion();
};

/*!
 * Application instance declaration 
 */

////@begin declare app
DECLARE_APP(wxparaverApp)
////@end declare app
