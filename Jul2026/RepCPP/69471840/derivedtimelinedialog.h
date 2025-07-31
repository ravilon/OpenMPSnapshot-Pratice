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
#include "wx/statline.h"
#include "wx/spinctrl.h"
////@end includes

#include "window.h"
#include "loadedwindows.h"

/*!
 * Forward declarations
 */

////@begin forward declarations
class wxSpinCtrl;
////@end forward declarations

/*!
 * Control identifiers
 */

////@begin control identifiers
#define ID_DERIVEDTIMELINEDIALOG 10032
#define ID_DERIVED_NAME 10001
#define ID_TOPCOMPOSE1 10002
#define ID_MINCOMPOSE1 10034
#define ID_MAXCOMPOSE1 10003
#define ID_TOPCOMPOSE2 10000
#define ID_MINCOMPOSE2 10041
#define ID_MAXCOMPOSE2 10040
#define ID_FACTOR_TIMELINE_1 10035
#define ID_SHIFT_TIMELINE1 10006
#define ID_TIMELINES_TEXT1 10036
#define ID_TIMELINES_BUTTON1 10004
#define ID_OPERATIONS 10037
#define ID_TIMELINES_TEXT2 10038
#define ID_TIMELINES_BUTTON2 10005
#define ID_SHIFT_TIMELINE2 10007
#define ID_FACTOR_TIMELINE_2 10039
#define ID_SWAP_WINDOWS 10033
#define SYMBOL_DERIVEDTIMELINEDIALOG_STYLE wxCAPTION|wxRESIZE_BORDER|wxSYSTEM_MENU|wxCLOSE_BOX|wxTAB_TRAVERSAL
#define SYMBOL_DERIVEDTIMELINEDIALOG_TITLE _("Create Derived Timeline Window")
#define SYMBOL_DERIVEDTIMELINEDIALOG_IDNAME ID_DERIVEDTIMELINEDIALOG
#define SYMBOL_DERIVEDTIMELINEDIALOG_SIZE wxDefaultSize
#define SYMBOL_DERIVEDTIMELINEDIALOG_POSITION wxDefaultPosition
////@end control identifiers

/*!
 * DerivedTimelineDialog class declaration
 */

class DerivedTimelineDialog: public wxDialog
{    
  DECLARE_DYNAMIC_CLASS( DerivedTimelineDialog )
  DECLARE_EVENT_TABLE()

public:
  /// Constructors
  DerivedTimelineDialog();
  DerivedTimelineDialog( wxWindow* parent, wxWindowID id = SYMBOL_DERIVEDTIMELINEDIALOG_IDNAME, const wxString& caption = SYMBOL_DERIVEDTIMELINEDIALOG_TITLE, const wxPoint& pos = SYMBOL_DERIVEDTIMELINEDIALOG_POSITION, const wxSize& size = SYMBOL_DERIVEDTIMELINEDIALOG_SIZE, long style = SYMBOL_DERIVEDTIMELINEDIALOG_STYLE );

  /// Creation
  bool Create( wxWindow* parent, wxWindowID id = SYMBOL_DERIVEDTIMELINEDIALOG_IDNAME, const wxString& caption = SYMBOL_DERIVEDTIMELINEDIALOG_TITLE, const wxPoint& pos = SYMBOL_DERIVEDTIMELINEDIALOG_POSITION, const wxSize& size = SYMBOL_DERIVEDTIMELINEDIALOG_SIZE, long style = SYMBOL_DERIVEDTIMELINEDIALOG_STYLE );

  /// Destructor
  ~DerivedTimelineDialog();

  /// Initialises member variables
  void Init();

  /// Creates the controls and sizers
  void CreateControls();

////@begin DerivedTimelineDialog event handler declarations

  /// wxEVT_COMMAND_CHOICE_SELECTED event handler for ID_TOPCOMPOSE1
  void OnTopcompose1Selected( wxCommandEvent& event );

  /// wxEVT_COMMAND_CHOICE_SELECTED event handler for ID_TOPCOMPOSE2
  void OnTopcompose2Selected( wxCommandEvent& event );

  /// wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_TIMELINES_BUTTON1
  void OnTimelinesButton1Click( wxCommandEvent& event );

  /// wxEVT_COMMAND_CHOICE_SELECTED event handler for ID_OPERATIONS
  void OnOperationsSelected( wxCommandEvent& event );

  /// wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_TIMELINES_BUTTON2
  void OnTimelinesButton2Click( wxCommandEvent& event );

  /// wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_SWAP_WINDOWS
  void OnSwapWindowsClick( wxCommandEvent& event );

  /// wxEVT_COMMAND_BUTTON_CLICKED event handler for wxID_OK
  void OnOkClick( wxCommandEvent& event );

////@end DerivedTimelineDialog event handler declarations

////@begin DerivedTimelineDialog member function declarations

  Timeline * GetCurrentWindow1() const { return currentWindow1 ; }
  void SetCurrentWindow1(Timeline * value) { currentWindow1 = value ; }

  Timeline * GetCurrentWindow2() const { return currentWindow2 ; }
  void SetCurrentWindow2(Timeline * value) { currentWindow2 = value ; }

  double GetFactorTimeline1() const { return factorTimeline1 ; }
  void SetFactorTimeline1(double value) { factorTimeline1 = value ; }

  double GetFactorTimeline2() const { return factorTimeline2 ; }
  void SetFactorTimeline2(double value) { factorTimeline2 = value ; }

  int GetLastTimeline1() const { return lastTimeline1 ; }
  void SetLastTimeline1(int value) { lastTimeline1 = value ; }

  int GetLastTimeline2() const { return lastTimeline2 ; }
  void SetLastTimeline2(int value) { lastTimeline2 = value ; }

  TParamValue GetMaxCompose1() const { return maxCompose1 ; }
  void SetMaxCompose1(TParamValue value) { maxCompose1 = value ; }

  TParamValue GetMaxCompose2() const { return maxCompose2 ; }
  void SetMaxCompose2(TParamValue value) { maxCompose2 = value ; }

  TParamValue GetMinCompose1() const { return minCompose1 ; }
  void SetMinCompose1(TParamValue value) { minCompose1 = value ; }

  TParamValue GetMinCompose2() const { return minCompose2 ; }
  void SetMinCompose2(TParamValue value) { minCompose2 = value ; }

  std::vector< std::string > GetOperations() const { return operations ; }
  void SetOperations(std::vector< std::string > value) { operations = value ; }

  PRV_INT16 GetShiftTimeline1() const { return shiftTimeline1 ; }
  void SetShiftTimeline1(PRV_INT16 value) { shiftTimeline1 = value ; }

  PRV_INT16 GetShiftTimeline2() const { return shiftTimeline2 ; }
  void SetShiftTimeline2(PRV_INT16 value) { shiftTimeline2 = value ; }

  std::string GetTimelineName() const { return timelineName ; }
  void SetTimelineName(std::string value) { timelineName = value ; }

  std::vector<TWindowID> GetTimelines1() const { return timelines1 ; }
  void SetTimelines1(std::vector<TWindowID> value) { timelines1 = value ; }

  std::vector<TWindowID> GetTimelines2() const { return timelines2 ; }
  void SetTimelines2(std::vector<TWindowID> value) { timelines2 = value ; }

  std::vector< std::string > GetTopCompose1() const { return topCompose1 ; }
  void SetTopCompose1(std::vector< std::string > value) { topCompose1 = value ; }

  std::vector< std::string > GetTopCompose2() const { return topCompose2 ; }
  void SetTopCompose2(std::vector< std::string > value) { topCompose2 = value ; }

  /// Retrieves bitmap resources
  wxBitmap GetBitmapResource( const wxString& name );

  /// Retrieves icon resources
  wxIcon GetIconResource( const wxString& name );
////@end DerivedTimelineDialog member function declarations

  /// Should we show tooltips?
  static bool ShowToolTips();

  bool TransferDataToWindow();
  bool TransferDataFromWindow();

////@begin DerivedTimelineDialog member variables
  wxTextCtrl* widgetName;
  wxChoice* widgetTopCompose1;
  wxStaticText* widgetLabelMinCompose1;
  wxTextCtrl* widgetMinCompose1;
  wxStaticText* widgetLabelMaxCompose1;
  wxTextCtrl* widgetMaxCompose1;
  wxChoice* widgetTopCompose2;
  wxStaticText* widgetLabelMinCompose2;
  wxTextCtrl* widgetMinCompose2;
  wxStaticText* widgetLabelMaxCompose2;
  wxTextCtrl* widgetMaxCompose2;
  wxStaticText* widgetLabelTimelines1;
  wxStaticText* widgetLabelTimelines2;
  wxTextCtrl* widgetFactorTimeline1;
  wxSpinCtrl* spinShiftTimeline1;
  wxTextCtrl* txtTimelines1;
  wxBitmapButton* buttonTimelines1;
  wxChoice* widgetOperations;
  wxTextCtrl* txtTimelines2;
  wxBitmapButton* buttonTimelines2;
  wxSpinCtrl* spinShiftTimeline2;
  wxTextCtrl* widgetFactorTimeline2;
  wxButton* swapWindowsButton;
private:
  Timeline * currentWindow1;
  Timeline * currentWindow2;
  double factorTimeline1;
  double factorTimeline2;
  int lastTimeline1;
  int lastTimeline2;
  TParamValue maxCompose1;
  TParamValue maxCompose2;
  TParamValue minCompose1;
  TParamValue minCompose2;
  std::vector< std::string > operations;
  PRV_INT16 shiftTimeline1;
  PRV_INT16 shiftTimeline2;
  std::string timelineName;
  std::vector<TWindowID> timelines1;
  std::vector<TWindowID> timelines2;
  std::vector< std::string > topCompose1;
  std::vector< std::string > topCompose2;
////@end DerivedTimelineDialog member variables

  void presetTimelineComboBox( std::vector< Timeline * > timelines,
                               Timeline *currentWindow,
                               wxComboBox *comboBox,
                               int& currentSelection );
  void presetStringChoiceBox( std::vector< std::string > list, wxChoice *choiceBox );
  void presetFactorField( double value, wxTextCtrl *field );
  void presetNameField( std::string whichName, wxTextCtrl *field );

  void getSelectedString( wxChoice *choiceBox, std::vector< std::string > &selection ) const;
  void getSelectedWindow( wxComboBox *comboBox, std::vector< Timeline * > &selection ) const;
  void getName( wxTextCtrl *field, std::string &whichName ) const;
  bool getFactorFields( double &whichFactor1,
                        double &whichFactor2 );
  void setParametersCompose( PRV_UINT32 compose,
                             std::string nameFunction,
                             PRV_UINT32 numParameters,
                             std::vector< std::string > namesParameters,
                             std::vector< std::vector< double > > defaultValues );

  bool getParameterCompose( wxTextCtrl *field,
                            TParamValue &parameter,
                            wxString prefixMessage );

  void setParameterComposeField( TParamValue defaultValues, wxTextCtrl *field );
  bool getParameterComposeField( wxTextCtrl *field, TParamValue &values );

};
