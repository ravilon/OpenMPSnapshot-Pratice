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

/// Create a text label
Y_API yTextLabel* Y_CALL yTextLabel_Create(yStringView text, uint32_t colour, yVector3 pos, float drawDist, int vw, bool los);

/// Destroy a text label
Y_API void Y_CALL yTextLabel_Destroy(yTextLabel* textLabel);

/// Get the label's position
Y_API yQuat Y_CALL yTextLabel_GetPosition(yTextLabel const* textLabel);

/// Set the label's position
Y_API void Y_CALL yTextLabel_SetPosition(yTextLabel* textLabel, yVector3 position);

/// Get the label's rotation
Y_API yQuat Y_CALL yTextLabel_GetRotation(yTextLabel const* textLabel);

/// Set the label's rotation
Y_API void Y_CALL yTextLabel_SetRotation(yTextLabel* textLabel, yQuat rotation);

/// Get the label's virtual world
Y_API int Y_CALL yTextLabel_GetVirtualWorld(yTextLabel const* textLabel);

/// Set the label's virtual world
Y_API void Y_CALL yTextLabel_SetVirtualWorld(yTextLabel* textLabel, int vw);

/// Set the text label's text
Y_API void Y_CALL yTextLabel_SetText(yTextLabel* textLabel, yStringView text);

/// Get the text label's text
Y_API yStringView Y_CALL yTextLabel_GetText(yTextLabel const* textLabel);

/// Set the text label's colour
Y_API void Y_CALL yTextLabel_SetColour(yTextLabel* textLabel, uint32_t colour);

/// Get the text label's colour
Y_API uint32_t Y_CALL yTextLabel_GetColour(yTextLabel const* textLabel);

/// Set the text label's draw distance
Y_API void Y_CALL yTextLabel_SetDrawDistance(yTextLabel* textLabel, float dist);

/// Get the text label's draw distance
Y_API float Y_CALL yTextLabel_GetDrawDistance(yTextLabel* textLabel);

/// Attach the text label to a player with an offset
Y_API void Y_CALL yTextLabel_AttachToPlayer(yTextLabel* textLabel, yPlayer* player, yVector3 offset);

/// Attach the text label to a vehicle with an offset
Y_API void Y_CALL yTextLabel_AttachToVehicle(yTextLabel* textLabel, yVehicle* vehicle, yVector3 offset);

/// Detach the text label from the player and set its position or offset
Y_API void Y_CALL yTextLabel_DetachFromPlayer(yTextLabel* textLabel, yVector3 position);

/// Detach the text label from the vehicle and set its position or offset
Y_API void Y_CALL yTextLabel_DetachFromVehicle(yTextLabel* textLabel, yVector3 position);

/// Set the text label's los check.
Y_API void Y_CALL yTextLabel_SetTestLOS(yTextLabel* textLabel, bool status);

/// Get the text label's los check status.
Y_API bool Y_CALL yTextLabel_GetTestLOS(yTextLabel const* textLabel);

/// Used to update both colour and text with one single network packet being sent.
Y_API void Y_CALL yTextLabel_SetColourAndText(yTextLabel* textLabel, uint32_t colour, yStringView text);

/// Checks if player has the text label streamed in for themselves
Y_API bool Y_CALL yTextLabel_IsStreamedInForPlayer(yTextLabel const* textLabel, yPlayer const* player);

/// Streams in the text label for a specific player
Y_API void Y_CALL yTextLabel_StreamInForPlayer(yTextLabel* textLabel, yPlayer* player);

/// Streams out the text label for a specific player
Y_API void Y_CALL yTextLabel_StreamOutForPlayer(yTextLabel* textLabel, yPlayer* player);

/* ---------------------------------------------------------------- */

/// Create a player text label
Y_API yPlayerTextLabel* Y_CALL yPlayer_CreateTextLabel(yPlayer* player, yStringView text, uint32_t colour, yVector3 pos, float drawDist, bool los);

/// Destroy a player text label
Y_API void Y_CALL yPlayer_DestroyTextLabel(yPlayer* player, yPlayerTextLabel* textLabel);

/// Get the label's position
Y_API yVector3 Y_CALL yPlayerTextLabel_GetPosition(yPlayerTextLabel const* textLabel);

/// Set the label's position
Y_API void Y_CALL yPlayerTextLabel_SetPosition(yPlayerTextLabel* textLabel, yVector3 position);

/// Get the label's rotation
Y_API yQuat Y_CALL yPlayerTextLabel_GetRotation(yPlayerTextLabel const* textLabel);

/// Set the label's rotation
Y_API void Y_CALL yPlayerTextLabel_SetRotation(yPlayerTextLabel* textLabel, yQuat rotation);

/// Get the label's virtual world
Y_API int Y_CALL yPlayerTextLabel_GetVirtualWorld(yPlayerTextLabel const* textLabel);

/// Set the label's virtual world
Y_API void Y_CALL yPlayerTextLabel_SetVirtualWorld(yPlayerTextLabel* textLabel, int vw);

/// Set the text label's text
Y_API void Y_CALL yPlayerTextLabel_SetText(yPlayerTextLabel* textLabel, yStringView text);

/// Get the text label's text
Y_API yStringView Y_CALL yPlayerTextLabel_GetText(yPlayerTextLabel const* textLabel);

/// Set the text label's colour
Y_API void Y_CALL yPlayerTextLabel_SetColour(yPlayerTextLabel* textLabel, uint32_t colour);

/// Get the text label's colour
Y_API uint32_t Y_CALL yPlayerTextLabel_GetColour(yPlayerTextLabel const* textLabel);

/// Set the text label's draw distance
Y_API void Y_CALL yPlayerTextLabel_SetDrawDistance(yPlayerTextLabel* textLabel, float dist);

/// Get the text label's draw distance
Y_API float Y_CALL yPlayerTextLabel_GetDrawDistance(yPlayerTextLabel* textLabel);

/// Attach the text label to a player with an offset
Y_API void Y_CALL yPlayerTextLabel_AttachToPlayer(yPlayerTextLabel* textLabel, yPlayer* player, yVector3 offset);

/// Attach the text label to a vehicle with an offset
Y_API void Y_CALL yPlayerTextLabel_AttachToVehicle(yPlayerTextLabel* textLabel, yVehicle* vehicle, yVector3 offset);

/// Detach the text label from the player and set its position or offset
Y_API void Y_CALL yPlayerTextLabel_DetachFromPlayer(yPlayerTextLabel* textLabel, yVector3 position);

/// Detach the text label from the vehicle and set its position or offset
Y_API void Y_CALL yPlayerTextLabel_DetachFromVehicle(yPlayerTextLabel* textLabel, yVector3 position);

/// Set the text label's los check.
Y_API void Y_CALL yPlayerTextLabel_SetTestLOS(yPlayerTextLabel* textLabel, bool status);

/// Get the text label's los check status.
Y_API bool Y_CALL yPlayerTextLabel_GetTestLOS(yPlayerTextLabel const* textLabel);

/// Used to update both colour and text with one single network packet being sent.
Y_API void Y_CALL yPlayerTextLabel_SetColourAndText(yPlayerTextLabel* textLabel, uint32_t colour, yStringView text);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
