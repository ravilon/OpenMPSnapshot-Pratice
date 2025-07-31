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

// For compilers that support precompilation, includes "wx/wx.h".
#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif

#include "connection.h"
#include "wxparaverapp.h"

using namespace std;

wxConnectionBase *stServer::OnAcceptConnection( const wxString& topic )
{
  if( topic.Lower() == wxT( "wxparaver" ) )
    return new stConnection();
  else
    return nullptr;
}

bool stConnection::OnExecute( const wxString& WXUNUSED( topic ),
                              const void *data,
                              size_t WXUNUSED( size ),
                              wxIPCFormat WXUNUSED( format ) )
{
  wxString dataStr( wxString::FromUTF8( (char *)data ));
  static wxString tmpCommand;

  if( dataStr.IsEmpty() )
  {
    if( wxparaverApp::mainWindow )
      wxparaverApp::mainWindow->Raise();
  }
  else if( dataStr == wxT( "BEGIN" ) )
  {
    wxparaverApp::mainWindow->SetCanServeSignal( false );
    tmpCommand.Clear();
  }
  else if( dataStr == wxT( "END" ) )
  {
    wxparaverApp::mainWindow->SetCanServeSignal( true );
    wxCmdLineParser tmpLine( wxparaverApp::argumentsParseSyntax/*, tmpCommand*/ );
    tmpLine.SetCmdLine( tmpCommand );
    tmpLine.Parse();
    
    wxGetApp().ParseCommandLine( tmpLine );
  }
  else
  {
    tmpCommand += dataStr + wxT( " " );
  }

  return true;
}
