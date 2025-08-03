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
    void (Y_CALL* OnPlayerEnterGangZone)(yPlayer* player, yGangZoneBase* zone);
    void (Y_CALL* OnPlayerLeaveGangZone)(yPlayer* player, yGangZoneBase* zone);
    void (Y_CALL* OnPlayerClickGangZone)(yPlayer* player, yGangZoneBase* zone);
} yGangZoneEventHandler;

Y_API void Y_CALL yAddGangZoneEventHandler(yGangZoneEventHandler const* handler);

Y_API yGangZoneBase* Y_CALL yGangZone_Create(yVector4 position);
Y_API void Y_CALL yGangZone_Destroy(yGangZoneBase* gangZone);

/// Check if a gangzone is shown for player
Y_API bool Y_CALL yGangZone_IsShownForPlayer(yGangZoneBase const* gangZone, yPlayer const* player);

/// Check if a gangzone is flashing for player
Y_API bool Y_CALL yGangZone_IsFlashingForPlayer(yGangZoneBase const* gangZone, yPlayer const* player);

/// Show a gangzone for player
Y_API void Y_CALL yGangZone_ShowForPlayer(yGangZoneBase* gangZone, yPlayer* player, uint32_t colour);

/// Hide a gangzone for player
Y_API void Y_CALL yGangZone_HideForPlayer(yGangZoneBase* gangZone, yPlayer* player);

/// Flashing a gangzone for player
Y_API void Y_CALL yGangZone_FlashForPlayer(yGangZoneBase* gangZone, yPlayer* player, uint32_t colour);

/// Stop flashing a gangzone for player
Y_API void Y_CALL yGangZone_StopFlashForPlayer(yGangZoneBase* gangZone, yPlayer* player);

/// Get position of gangzone.
Y_API yVector4 Y_CALL yGangZone_GetPosition(yGangZoneBase const* gangZone);

/// Set position of gangzone.
Y_API void Y_CALL yGangZone_SetPosition(yGangZoneBase* gangZone, yVector4 position);

/// Check if specified player is within gangzone bounds (only works with IGangZonesComponent::useGangZoneCheck).
Y_API bool Y_CALL yGangZone_IsPlayerInside(yGangZoneBase const* gangZone, yPlayer const* player);

/// get gangzone flashing color for a player
Y_API uint32_t Y_CALL yGangZone_GetFlashingColourForPlayer(yGangZoneBase const* gangZone, yPlayer* player);

/// get gangzone color for a player
Y_API uint32_t Y_CALL yGangZone_GetColourForPlayer(yGangZoneBase const* gangZone, yPlayer* player);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
