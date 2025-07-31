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


#include <wx/treectrl.h>
#include <wx/choicebk.h>
#include <vector>
#include <string>

class gTimeline;
class gHistogram;
class paraverMain;
class Timeline;
class Histogram;
class Trace;

// gTimeline and gHistogram ancestor
// May absorb other methods.
class gWindow
{
  public:
    gWindow()
    {
      enableButtonDestroy = true;
    }

    bool getEnableDestroyButton() const { return enableButtonDestroy ; }
    virtual void setEnableDestroyButton( bool value ) { enableButtonDestroy = value ; }

  private:
    bool enableButtonDestroy;
};


wxTreeCtrl * createTree( wxImageList *imageList );
wxTreeCtrl *getAllTracesTree();
wxTreeCtrl *getSelectedTraceTree( Trace *trace );

void appendHistogram2Tree( gHistogram *ghistogram );

wxTreeItemId getItemIdFromWindow( wxTreeItemId root, Timeline *wanted, bool &found );
wxTreeItemId getItemIdFromGTimeline( wxTreeItemId root, gTimeline *wanted, bool &found );
gTimeline *getGTimelineFromWindow( wxTreeItemId root, Timeline *wanted, bool &found );
gHistogram *getGHistogramFromWindow( wxTreeItemId root, Histogram *wanted );
void getParentGTimeline( gTimeline *current, std::vector< gTimeline * > & children );

void BuildTree( paraverMain *parent,
                wxTreeCtrl *root1, wxTreeItemId idRoot1,
                wxTreeCtrl *root2, wxTreeItemId idRoot2,
                Timeline *window,
                std::string nameSuffix = std::string("") );

bool updateTreeItem( wxTreeCtrl *tree,
                     wxTreeItemId& id,
                     std::vector< Timeline * > &allWindows,
                     std::vector< Histogram * > &allHistograms,
                     wxWindow **currentWindow,
                     bool allTracesTree );

void iconizeWindows( wxTreeCtrl *tree,
                     wxTreeItemId& id,
                     bool iconize );

int getIconNumber( Timeline *whichWindow );


