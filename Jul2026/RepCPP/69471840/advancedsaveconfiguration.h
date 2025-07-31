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
#include "wx/tglbtn.h"
////@end includes

#include <vector>
#include <map>
#include <set>

#include "paraverkerneltypes.h"
#include "window.h"
#include "histogram.h"
#include "cfgs4d.h"

using std::multimap;

/*!
 * Forward declarations
 */

////@begin forward declarations
class wxToggleButton;
////@end forward declarations

/*!
 * Control identifiers
 */

////@begin control identifiers
#define ID_ADVANCEDSAVECONFIGURATION 10186
#define ID_CHOICE_WINDOW 10185
#define ID_SCROLLEDWINDOW1 10187
#define ID_SCROLLED_LINK_PROPERTIES 10000
#define ID_TOGGLEBUTTON_LIST_SELECTED 10197
#define SYMBOL_ADVANCEDSAVECONFIGURATION_STYLE wxCAPTION|wxRESIZE_BORDER|wxSYSTEM_MENU|wxCLOSE_BOX|wxTAB_TRAVERSAL
#define SYMBOL_ADVANCEDSAVECONFIGURATION_TITLE _("Save Basic CFG - Properties Editor")
#define SYMBOL_ADVANCEDSAVECONFIGURATION_IDNAME ID_ADVANCEDSAVECONFIGURATION
#define SYMBOL_ADVANCEDSAVECONFIGURATION_SIZE wxDefaultSize
#define SYMBOL_ADVANCEDSAVECONFIGURATION_POSITION wxDefaultPosition
////@end control identifiers


/*!
 * AdvancedSaveConfiguration class declaration
 */

enum class TEditorMode
{
  PROPERTIES_TAGS,
  HISTOGRAM_STATISTIC_TAGS
};

class AdvancedSaveConfiguration: public wxDialog
{    
  DECLARE_DYNAMIC_CLASS( AdvancedSaveConfiguration )
  DECLARE_EVENT_TABLE()

public:

  /// Constructors
  AdvancedSaveConfiguration();
  AdvancedSaveConfiguration( wxWindow* parent,
                             wxWindowID id = SYMBOL_ADVANCEDSAVECONFIGURATION_IDNAME,
                             const wxString& caption = SYMBOL_ADVANCEDSAVECONFIGURATION_TITLE,
                             const wxPoint& pos = SYMBOL_ADVANCEDSAVECONFIGURATION_POSITION,
                             const wxSize& size = SYMBOL_ADVANCEDSAVECONFIGURATION_SIZE,
                             long style = SYMBOL_ADVANCEDSAVECONFIGURATION_STYLE );
  AdvancedSaveConfiguration( wxWindow* parent,
                             const std::vector< Timeline * > &whichTimelines,
                             const std::vector< Histogram * > &whichHistograms,
                             TEditorMode mode = TEditorMode::PROPERTIES_TAGS,
                             wxWindowID id = SYMBOL_ADVANCEDSAVECONFIGURATION_IDNAME,
                             const wxString& caption = SYMBOL_ADVANCEDSAVECONFIGURATION_TITLE,
                             const wxPoint& pos = SYMBOL_ADVANCEDSAVECONFIGURATION_POSITION,
                             const wxSize& size = SYMBOL_ADVANCEDSAVECONFIGURATION_SIZE,
                             long style = SYMBOL_ADVANCEDSAVECONFIGURATION_STYLE );

  /// Creation
  bool Create( wxWindow* parent, wxWindowID id = SYMBOL_ADVANCEDSAVECONFIGURATION_IDNAME, const wxString& caption = SYMBOL_ADVANCEDSAVECONFIGURATION_TITLE, const wxPoint& pos = SYMBOL_ADVANCEDSAVECONFIGURATION_POSITION, const wxSize& size = SYMBOL_ADVANCEDSAVECONFIGURATION_SIZE, long style = SYMBOL_ADVANCEDSAVECONFIGURATION_STYLE );

  /// Destructor
  ~AdvancedSaveConfiguration();

  /// Initialises member variables
  void Init();

  /// Creates the controls and sizers
  void CreateControls();

////@begin AdvancedSaveConfiguration event handler declarations

  /// wxEVT_COMMAND_CHOICE_SELECTED event handler for ID_CHOICE_WINDOW
  void OnChoiceWindowSelected( wxCommandEvent& event );

  /// wxEVT_COMMAND_CHECKBOX_CLICKED event handler for ID_TOGGLEBUTTON_LIST_SELECTED
  void OnToggleOnlySelectedClick( wxCommandEvent& event );

  /// wxEVT_COMMAND_BUTTON_CLICKED event handler for wxID_CANCEL
  void OnCancelClick( wxCommandEvent& event );

  /// wxEVT_COMMAND_BUTTON_CLICKED event handler for wxID_SAVE
  void OnSaveClick( wxCommandEvent& event );

////@end AdvancedSaveConfiguration event handler declarations

////@begin AdvancedSaveConfiguration member function declarations

  /// Retrieves bitmap resources
  wxBitmap GetBitmapResource( const wxString& name );

  /// Retrieves icon resources
  wxIcon GetIconResource( const wxString& name );
////@end AdvancedSaveConfiguration member function declarations

  const CFGS4DLinkedPropertiesManager& getLinkedPropertiesManager() const;

  /// Should we show tooltips?
  static bool ShowToolTips();

////@begin AdvancedSaveConfiguration member variables
  wxChoice* choiceWindow;
  wxScrolledWindow* scrolledWindow;
  wxScrolledWindow* scrolledLinkProperties;
  wxToggleButton* toggleOnlySelected;
  wxButton* buttonSave;
////@end AdvancedSaveConfiguration member variables


  protected:
    const static wxString KParamSeparator;
    const static wxString KPreffixSeparator;
    const static wxString KSuffixSeparator;
    const static wxString KCheckBoxSuffix;
    const static wxString KTextCtrlSuffix;
    const static wxString KButtonSuffix;

  private:
    bool isTimeline;
    int currentItem;                   // Index to selected item in choice widget. Used to compute
                                       // position of Timeline * or Histogram * in next vectors.
    std::vector< Timeline * > timelines;
    std::vector< Histogram * > histograms;

    std::map< Timeline *, bool > backupTimelinesCFG4DEnabled;
    std::map< Timeline *, bool > backupTimelinesCFG4DMode;
    std::map< Timeline *, std::map< std::string, std::string > > backupTimelinesCFG4DAliasList;
    std::map< Timeline *, Timeline::TParamAlias > backupTimelinesCFG4DParamAlias;
    std::map< Histogram *, bool > backupHistogramsCFG4DEnabled;
    std::map< Histogram *, bool > backupHistogramsCFG4DMode;
    std::map< Histogram *, std::map< std::string, std::string > > backupHistogramsCFG4DAliasList;
    std::map< Histogram *, std::map< std::string, std::string > > backupHistogramsCFG4DStatisticsAliasList;

    std::vector< std::string > fullTagList;
    std::map< std::string, std::string > renamedTag;
    std::map< std::string, bool > enabledTag;
    bool enabledCFG4DMode;

    TEditorMode editionMode;

    CFGS4DLinkedPropertiesManager unlinkedManager;
    CFGS4DLinkedPropertiesManager linkedManager;

    int GetSelectionIndexCorrected( int index, bool &isTimeline );

    void initLinks( Timeline *whichTimeline );
    void initLinks( Histogram *whichHistogram );

    wxString BuildName( Timeline *current );
    wxString BuildName( Histogram *current );

    bool allowedLevel( const std::string &tag );

    void BuildTagMaps( const std::map< std::string, std::string > &renamedTagMap,
                       const bool showFullList );
    void parseSemanticParameterTag( const wxString& whichTag,
                                    std::string& onSemanticLevel,
                                    std::string& onFunction,
                                    TParamIndex& onNumParameter );
    void InsertParametersToTagMaps( const std::vector< Timeline::TParamAliasKey > &fullParamList,
                                    const Timeline::TParamAlias &renamedParamAlias,
                                    const bool showFullList );
    wxBoxSizer *BuildTagRowWidgets( std::map< std::string, std::string >::iterator it,
                                    bool showFullList );
    void BuildTagWidgets( const bool showFullList );
    void BuildTagsPanel( Timeline *currentWindow, const bool showFullList );
    void BuildTagsPanel( Histogram *currentHistogram, const bool showFullList );

    void PreparePanel( bool showFullList );
    void TransferDataFromPanel( bool showFullList );

    void DisconnectWidgetsTagsPanel( bool showFullList );
    void CleanTagsPanel( bool showFullList );

    wxCheckBox *GetCheckBoxByName( const wxString& widgetName ) const;
    wxTextCtrl *GetTextCtrlByName( const wxString& widgetName ) const;
    wxButton *GetButtonByName( const wxString& widgetName ) const;

    template< class T >
    void insertLinkInUnlinkedManager( const std::string& originalName, const std::string& newCustomName, T *whichWindow );

    void OnCheckBoxPropertyClicked( wxCommandEvent& event );
    void OnCheckBoxLinkWindowClicked( wxCommandEvent& event );
    void OnStatisticsButtonClick( wxCommandEvent& event );
    void OnTextCtrlPropertyChanged( wxCommandEvent &event );

    void RefreshList( bool showFullList );

    void buildWindowsSetWidgets( const std::string& propertyName, wxBoxSizer *boxSizerLinks, bool checked );
    template <typename WindowType>
    void buildLinkWindowWidget( wxBoxSizer *boxSizerLinks, 
                                const std::string& propertyName,
                                WindowType *whichWindow,
                                bool checked );
    void updateLinkPropertiesWidgets();
    void updateAliasForLinkedWindows( std::string whichOriginalName, std::string whichCustomName );

    void OnCheckBoxLinkPropertyClicked( wxCommandEvent& event );
    void OnLinkedPropertiesNameChanged( wxCommandEvent& event );

    void setTimelineCFG4DAlias( Timeline *whichWindow,
                                const std::string& whichOriginalName,
                                const std::string& whichCustomName );

};
