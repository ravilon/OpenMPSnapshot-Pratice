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
void (Y_CALL* OnPlayerEnterCheckpoint)(yPlayer* player);
void (Y_CALL* OnPlayerLeaveCheckpoint)(yPlayer* player);
void (Y_CALL* OnPlayerEnterRaceCheckpoint)(yPlayer* player);
void (Y_CALL* OnPlayerLeaveRaceCheckpoint)(yPlayer* player);
} yPlayerCheckpointEventHandler;

Y_API void Y_CALL yAddCheckpointEventHandler(yPlayerCheckpointEventHandler const* handler);

Y_API yVector3 Y_CALL yPlayer_GetCheckpointPosition(yPlayer* player);
Y_API void Y_CALL yPlayer_SetCheckpointPosition(yPlayer* player, yVector3 position);
Y_API float Y_CALL yPlayer_GetCheckpointRadius(yPlayer* player);
Y_API void Y_CALL yPlayer_SetCheckpointRadius(yPlayer* player, float radius);
Y_API bool Y_CALL yPlayer_IsPlayerInsideCheckpoint(yPlayer* player);
Y_API void Y_CALL yPlayer_SetPlayerInsideCheckpoint(yPlayer* player, bool inside);
Y_API void Y_CALL yPlayer_EnableCheckpoint(yPlayer* player);
Y_API void Y_CALL yPlayer_DisableCheckpoint(yPlayer* player);
Y_API bool Y_CALL yPlayer_IsEnabledCheckpoint(yPlayer* player);

Y_API yVector3 Y_CALL yPlayer_GetRaceCheckpointPosition(yPlayer* player);
Y_API void Y_CALL yPlayer_SetRaceCheckpointPosition(yPlayer* player, yVector3 position);
Y_API float Y_CALL yPlayer_GetCheckpointRadiusRace(yPlayer* player);
Y_API void Y_CALL yPlayer_SetRaceCheckpointRadius(yPlayer* player, float radius);
Y_API bool Y_CALL yPlayer_IsPlayerInsideRaceCheckpoint(yPlayer* player);
Y_API void Y_CALL yPlayer_SetPlayerInsideRaceCheckpoint(yPlayer* player, bool inside);
Y_API void Y_CALL yPlayer_EnableRaceCheckpoint(yPlayer* player);
Y_API void Y_CALL yPlayer_DisableRaceCheckpoint(yPlayer* player);
Y_API bool Y_CALL yPlayer_IsEnabledRaceCheckpoint(yPlayer* player);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
