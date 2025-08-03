/*-------------------------------------------------------------------
 *  Copyright (c) 2025 Maicol Castro <maicolcastro.abc@gmail.com>.
 *  All rights reserved.
 *
 *  Distributed under the BSD 3-Clause License.
 *  See LICENSE.txt in the root directory of this project or at
 *  https://opensource.org/license/bsd-3-clause.
 *-----------------------------------------------------------------*/

#pragma once

#include "Common.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef enum {
    Y_TEXT_DRAW_ALIGNMENT_DEFAULT,
    Y_TEXT_DRAW_ALIGNMENT_LEFT,
    Y_TEXT_DRAW_ALIGNMENT_CENTER,
    Y_TEXT_DRAW_ALIGNMENT_RIGHT
} yTextDrawAlignmentTypes;

typedef enum {
    Y_TEXT_DRAW_STYLE_0,
    Y_TEXT_DRAW_STYLE_1,
    Y_TEXT_DRAW_STYLE_2,
    Y_TEXT_DRAW_STYLE_3,
    Y_TEXT_DRAW_STYLE_4,
    Y_TEXT_DRAW_STYLE_5,
    Y_TEXT_DRAW_STYLE_FONT_BECKETT_REGULAR = 0,
    Y_TEXT_DRAW_STYLE_FONT_AHARONI_BOLD,
    Y_TEXT_DRAW_STYLE_FONT_BANK_GOTHIC,
    Y_TEXT_DRAW_STYLE_FONT_PRICEDOWN,
    Y_TEXT_DRAW_STYLE_SPRITE,
    Y_TEXT_DRAW_STYLE_PREVIEW
} yTextDrawStyle;

typedef struct {
    void (Y_CALL* OnPlayerClickTextDraw)(yPlayer* player, yTextDraw* textDraw);
    void (Y_CALL* OnPlayerClickPlayerTextDraw)(yPlayer* player, yPlayerTextDraw* textDraw);
    bool (Y_CALL* OnPlayerCancelTextDrawSelection)(yPlayer* player);
    bool (Y_CALL* OnPlayerCancelPlayerTextDrawSelection)(yPlayer* player);
} yTextDrawEventHandler;

Y_API void Y_CALL yAddTextDrawEventHandler(yTextDrawEventHandler const* handler);

/// Create a new textdraw with some text
Y_API yTextDraw* Y_CALL yTextDraw_Create(yVector2 position, yStringView text);

/// Create a new textdraw with some preview model
Y_API yTextDraw* Y_CALL yTextDraw_CreatePreview(yVector2 position, int model);

/// Destroy a textdraw
Y_API void Y_CALL yTextDraw_Destroy(yTextDraw* textDraw);

/// Get the textdraw's position
Y_API yVector2 Y_CALL yTextDraw_GetPosition(yTextDraw const* textDraw);

/// Set the textdraw's position
Y_API void Y_CALL yTextDraw_SetPosition(yTextDraw* textDraw, yVector2 position);

/// Set the textdraw's text
Y_API void Y_CALL yTextDraw_SetText(yTextDraw* textDraw, yStringView text);

/// Get the textdraw's text
Y_API yStringView Y_CALL yTextDraw_GetText(yTextDraw const* textDraw);

/// Set the letter size
Y_API void Y_CALL yTextDraw_SetLetterSize(yTextDraw* textDraw, yVector2 size);

/// Get the letter size
Y_API yVector2 Y_CALL yTextDraw_GetLetterSize(yTextDraw const* textDraw);

/// Set the text size
Y_API void Y_CALL yTextDraw_SetTextSize(yTextDraw* textDraw, yVector2 size);

/// Get the text size
Y_API yVector2 Y_CALL yTextDraw_GetTextSize(yTextDraw const* textDraw);

/// Set the text alignment
Y_API void Y_CALL yTextDraw_SetAlignment(yTextDraw* textDraw, yTextDrawAlignmentTypes alignment);

/// Get the text alignment
Y_API yTextDrawAlignmentTypes Y_CALL yTextDraw_GetAlignment(yTextDraw const* textDraw);

/// Set the letters' colour
Y_API void Y_CALL yTextDraw_SetColour(yTextDraw* textDraw, uint32_t colour);

/// Get the letters' colour
Y_API uint32_t Y_CALL yTextDraw_GetLetterColour(yTextDraw const* textDraw);

/// Set whether the textdraw uses a box
Y_API void Y_CALL yTextDraw_UseBox(yTextDraw* textDraw, bool use);

/// Get whether the textdraw uses a box
Y_API bool Y_CALL yTextDraw_HasBox(yTextDraw const* textDraw);

/// Set the textdraw box's colour
Y_API void Y_CALL yTextDraw_SetBoxColour(yTextDraw* textDraw, uint32_t colour);

/// Get the textdraw box's colour
Y_API uint32_t Y_CALL yTextDraw_GetBoxColour(yTextDraw const* textDraw);

/// Set the textdraw's shadow strength
Y_API void Y_CALL yTextDraw_SetShadow(yTextDraw* textDraw, int shadow);

/// Get the textdraw's shadow strength
Y_API int Y_CALL yTextDraw_GetShadow(yTextDraw const* textDraw);

/// Set the textdraw's outline
Y_API void Y_CALL yTextDraw_SetOutline(yTextDraw* textDraw, int outline);

/// Get the textdraw's outline
Y_API int Y_CALL yTextDraw_GetOutline(yTextDraw const* textDraw);

/// Set the textdraw's background colour
Y_API void Y_CALL yTextDraw_SetBackgroundColour(yTextDraw* textDraw, uint32_t colour);

/// Get the textdraw's background colour
Y_API uint32_t Y_CALL yTextDraw_GetBackgroundColour(yTextDraw const* textDraw);

/// Set the textdraw's drawing style
Y_API void Y_CALL yTextDraw_SetStyle(yTextDraw* textDraw, yTextDrawStyle style);

/// Get the textdraw's drawing style
Y_API yTextDrawStyle Y_CALL yTextDraw_GetStyle(yTextDraw const* textDraw);

/// Set whether the textdraw is proportional
Y_API void Y_CALL yTextDraw_SetProportional(yTextDraw* textDraw, bool proportional);

/// Get whether the textdraw is proportional
Y_API bool Y_CALL yTextDraw_IsProportional(yTextDraw const* textDraw);

/// Set whether the textdraw is selectable
Y_API void Y_CALL yTextDraw_SetSelectable(yTextDraw* textDraw, bool selectable);

/// Get whether the textdraw is selectable
Y_API bool Y_CALL yTextDraw_IsSelectable(yTextDraw const* textDraw);

/// Set the textdraw's preview model
Y_API void Y_CALL yTextDraw_SetPreviewModel(yTextDraw* textDraw, int model);

/// Get the textdraw's preview model
Y_API int Y_CALL yTextDraw_GetPreviewModel(yTextDraw const* textDraw);

/// Set the textdraw's preview rotation
Y_API void Y_CALL yTextDraw_SetPreviewRotation(yTextDraw* textDraw, yVector3 rotation);

/// Get the textdraw's preview rotation
Y_API yVector3 Y_CALL yTextDraw_GetPreviewRotation(yTextDraw const* textDraw);

/// Set the textdraw's preview vehicle colours
Y_API void Y_CALL yTextDraw_SetPreviewVehicleColour(yTextDraw* textDraw, int colour1, int colour2);

/// Get the textdraw's preview vehicle colours
Y_API void Y_CALL yTextDraw_GetPreviewVehicleColour(yTextDraw const* textDraw, int* colour1, int* colour2);

/// Set the textdraw's preview zoom factor
Y_API void Y_CALL yTextDraw_SetPreviewZoom(yTextDraw* textDraw, float zoom);

/// Get the textdraw's preview zoom factor
Y_API float Y_CALL yTextDraw_GetPreviewZoom(yTextDraw const* textDraw);

/// Restream the textdraw
Y_API void Y_CALL yTextDraw_Restream(yTextDraw* textDraw);

/// Show the textdraw for a player
Y_API void Y_CALL yTextDraw_ShowForPlayer(yTextDraw* textDraw, yPlayer* player);

/// Hide the textdraw for a player
Y_API void Y_CALL yTextDraw_HideForPlayer(yTextDraw* textDraw, yPlayer* player);

/// Get whether the textdraw is shown for a player
Y_API bool Y_CALL yTextDraw_IsShownForPlayer(yTextDraw const* textDraw, yPlayer const* player);

/// Set the textdraw's text for one player
Y_API void Y_CALL yTextDraw_SetTextForPlayer(yTextDraw* textDraw, yPlayer* player, yStringView text);

/* ---------------------------------------------------------------- */

/// Create a new textdraw with some text
Y_API yPlayerTextDraw* Y_CALL yPlayer_CreateTextDraw(yPlayer* player, yVector2 position, yStringView text);

/// Create a new textdraw with some preview model
Y_API yPlayerTextDraw* Y_CALL yPlayer_CreateTextDrawPreview(yPlayer* player, yVector2 position, int model);

/// Destroy a player textdraw
Y_API void Y_CALL yPlayer_DestroyTextDraw(yPlayer* player, yPlayerTextDraw* textDraw);

/// Get the textdraw's position
Y_API yVector2 Y_CALL yPlayerTextDraw_GetPosition(yPlayerTextDraw const* textDraw);

/// Set the textdraw's position
Y_API void Y_CALL yPlayerTextDraw_SetPosition(yPlayerTextDraw* textDraw, yVector2 position);

/// Set the textdraw's text
Y_API void Y_CALL yPlayerTextDraw_SetText(yPlayerTextDraw* textDraw, yStringView text);

/// Get the textdraw's text
Y_API yStringView Y_CALL yPlayerTextDraw_GetText(yPlayerTextDraw const* textDraw);

/// Set the letter size
Y_API void Y_CALL yPlayerTextDraw_SetLetterSize(yPlayerTextDraw* textDraw, yVector2 size);

/// Get the letter size
Y_API yVector2 Y_CALL yPlayerTextDraw_GetLetterSize(yPlayerTextDraw const* textDraw);

/// Set the text size
Y_API void Y_CALL yPlayerTextDraw_SetTextSize(yPlayerTextDraw* textDraw, yVector2 size);

/// Get the text size
Y_API yVector2 Y_CALL yPlayerTextDraw_GetTextSize(yPlayerTextDraw const* textDraw);

/// Set the text alignment
Y_API void Y_CALL yPlayerTextDraw_SetAlignment(yPlayerTextDraw* textDraw, yTextDrawAlignmentTypes alignment);

/// Get the text alignment
Y_API yTextDrawAlignmentTypes Y_CALL yPlayerTextDraw_GetAlignment(yPlayerTextDraw const* textDraw);

/// Set the letters' colour
Y_API void Y_CALL yPlayerTextDraw_SetColour(yPlayerTextDraw* textDraw, uint32_t colour);

/// Get the letters' colour
Y_API uint32_t Y_CALL yPlayerTextDraw_GetLetterColour(yPlayerTextDraw const* textDraw);

/// Set whether the textdraw uses a box
Y_API void Y_CALL yPlayerTextDraw_UseBox(yPlayerTextDraw* textDraw, bool use);

/// Get whether the textdraw uses a box
Y_API bool Y_CALL yPlayerTextDraw_HasBox(yPlayerTextDraw const* textDraw);

/// Set the textdraw box's colour
Y_API void Y_CALL yPlayerTextDraw_SetBoxColour(yPlayerTextDraw* textDraw, uint32_t colour);

/// Get the textdraw box's colour
Y_API uint32_t Y_CALL yPlayerTextDraw_GetBoxColour(yPlayerTextDraw const* textDraw);

/// Set the textdraw's shadow strength
Y_API void Y_CALL yPlayerTextDraw_SetShadow(yPlayerTextDraw* textDraw, int shadow);

/// Get the textdraw's shadow strength
Y_API int Y_CALL yPlayerTextDraw_GetShadow(yPlayerTextDraw const* textDraw);

/// Set the textdraw's outline
Y_API void Y_CALL yPlayerTextDraw_SetOutline(yPlayerTextDraw* textDraw, int outline);

/// Get the textdraw's outline
Y_API int Y_CALL yPlayerTextDraw_GetOutline(yPlayerTextDraw const* textDraw);

/// Set the textdraw's background colour
Y_API void Y_CALL yPlayerTextDraw_SetBackgroundColour(yPlayerTextDraw* textDraw, uint32_t colour);

/// Get the textdraw's background colour
Y_API uint32_t Y_CALL yPlayerTextDraw_GetBackgroundColour(yPlayerTextDraw const* textDraw);

/// Set the textdraw's drawing style
Y_API void Y_CALL yPlayerTextDraw_SetStyle(yPlayerTextDraw* textDraw, yTextDrawStyle style);

/// Get the textdraw's drawing style
Y_API yTextDrawStyle Y_CALL yPlayerTextDraw_GetStyle(yPlayerTextDraw const* textDraw);

/// Set whether the textdraw is proportional
Y_API void Y_CALL yPlayerTextDraw_SetProportional(yPlayerTextDraw* textDraw, bool proportional);

/// Get whether the textdraw is proportional
Y_API bool Y_CALL yPlayerTextDraw_IsProportional(yPlayerTextDraw const* textDraw);

/// Set whether the textdraw is selectable
Y_API void Y_CALL yPlayerTextDraw_SetSelectable(yPlayerTextDraw* textDraw, bool selectable);

/// Get whether the textdraw is selectable
Y_API bool Y_CALL yPlayerTextDraw_IsSelectable(yPlayerTextDraw const* textDraw);

/// Set the textdraw's preview model
Y_API void Y_CALL yPlayerTextDraw_SetPreviewModel(yPlayerTextDraw* textDraw, int model);

/// Get the textdraw's preview model
Y_API int Y_CALL yPlayerTextDraw_GetPreviewModel(yPlayerTextDraw const* textDraw);

/// Set the textdraw's preview rotation
Y_API void Y_CALL yPlayerTextDraw_SetPreviewRotation(yPlayerTextDraw* textDraw, yVector3 rotation);

/// Get the textdraw's preview rotation
Y_API yVector3 Y_CALL yPlayerTextDraw_GetPreviewRotation(yPlayerTextDraw const* textDraw);

/// Set the textdraw's preview vehicle colours
Y_API void Y_CALL yPlayerTextDraw_SetPreviewVehicleColour(yPlayerTextDraw* textDraw, int colour1, int colour2);

/// Get the textdraw's preview vehicle colours
Y_API void Y_CALL yPlayerTextDraw_GetPreviewVehicleColour(yPlayerTextDraw const* textDraw, int* colour1, int* colour2);

/// Set the textdraw's preview zoom factor
Y_API void Y_CALL yPlayerTextDraw_SetPreviewZoom(yPlayerTextDraw* textDraw, float zoom);

/// Get the textdraw's preview zoom factor
Y_API float Y_CALL yPlayerTextDraw_GetPreviewZoom(yPlayerTextDraw const* textDraw);

/// Restream the textdraw
Y_API void Y_CALL yPlayerTextDraw_Restream(yPlayerTextDraw* textDraw);

/// Show the textdraw for its player
Y_API void Y_CALL yPlayerTextDraw_Show(yPlayerTextDraw* textDraw);

/// Hide the textdraw for its player
Y_API void Y_CALL yPlayerTextDraw_Hide(yPlayerTextDraw* textDraw);

/// Get whether the textdraw is shown for its player
Y_API bool Y_CALL yPlayerTextDraw_IsShown(yPlayerTextDraw* textDraw);

/// Begin selecting textdraws for the player
Y_API void Y_CALL yPlayer_BeginSelection(yPlayer* player, uint32_t highlight);

/// Get whether the player is selecting textdraws
Y_API bool Y_CALL yPlayer_IsSelecting(yPlayer* player);

/// Stop selecting textdraws for the player
Y_API void Y_CALL yPlayer_EndSelection(yPlayer* player);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
