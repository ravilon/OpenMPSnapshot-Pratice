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

typedef struct {
    void (Y_CALL* OnPlayerPickUpPickup)(yPlayer* player, yPickupBase* pickup);
} yPickupEventHandler;

Y_API void Y_CALL yPickup_AddEventHandler(yPickupEventHandler const* handler);

/// Create a pickup
Y_API yPickupBase* Y_CALL yPickup_Create(int modelId, uint8_t type, yVector3 pos, uint32_t virtualWorld, bool isStatic);

/// Destroy a pickup
Y_API void Y_CALL yPickup_Destroy(yPickupBase* pickup);

/// Get the pickup's position
Y_API yVector3 Y_CALL yPickup_GetPosition(yPickupBase const* pickup);

/// Set the pickup's position
Y_API void Y_CALL yPickup_SetPosition(yPickupBase* pickup, yVector3 position);

/// Get the pickup's rotation
Y_API yQuat Y_CALL yPickup_GetRotation(yPickupBase const* pickup);

/// Set the pickup's rotation
Y_API void Y_CALL yPickup_SetRotation(yPickupBase* pickup, yQuat rotation);

/// Get the pickup's virtual world
Y_API int Y_CALL yPickup_GetVirtualWorld(yPickupBase const* pickup);

/// Set the pickup's virtual world
Y_API void Y_CALL yPickup_SetVirtualWorld(yPickupBase* pickup, int vw);

/// Sets pickup's type and restreams
Y_API void Y_CALL yPickup_SetType(yPickupBase* pickup, uint8_t type, bool update);

/// Gets pickup's type
Y_API uint8_t Y_CALL yPickup_GetType(yPickupBase* pickup);

/// Sets pickup's position but don't restream
Y_API void Y_CALL yPickup_SetPositionNoUpdate(yPickupBase* pickup, yVector3 position);

/// Sets pickup's model and restreams
Y_API void Y_CALL yPickup_SetModel(yPickupBase* pickup, int id, bool update);

/// Gets pickup's model
Y_API int Y_CALL yPickup_GetModel(yPickupBase* pickup);

/// Checks if pickup is streamed for a player
Y_API bool Y_CALL yPickup_IsStreamedInForPlayer(yPickupBase* pickup, yPlayer const* player);

/// Streams pickup for a player
Y_API void Y_CALL yPickup_StreamInForPlayer(yPickupBase* pickup, yPlayer* player);

/// Streams out pickup for a player
Y_API void Y_CALL yPickup_StreamOutForPlayer(yPickupBase* pickup, yPlayer* player);

/// Set pickup state hidden or shown for a player (only process streaming if pickup is not hidden)
Y_API void Y_CALL yPickup_SetPickupHiddenForPlayer(yPickupBase* pickup, yPlayer* player, bool hidden);

/// Check if given pickup has hidden state for player (only process streaming if pickup is not hidden)
Y_API bool Y_CALL yPickup_IsPickupHiddenForPlayer(yPickupBase* pickup, yPlayer* player);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
