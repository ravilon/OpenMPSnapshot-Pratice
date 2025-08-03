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
    int team;
    int skin;
    yVector3 spawn;
    float angle;
    yWeaponSlotData weapons[Y_MAX_WEAPON_SLOTS];
} yClassData;

typedef struct {
    bool (Y_CALL* OnPlayerRequestClass)(yPlayer* player, unsigned int classId);
} yClassEventHandler;

Y_API void Y_CALL yAddClassEventHandler(yClassEventHandler const* handler);

Y_API yClass* Y_CALL yClass_Create(yClassData const* data);
Y_API void Y_CALL yClass_Destroy(yClass* klass);
Y_API void Y_CALL yClass_Update(yClass* klass, yClassData const* data);

Y_API void Y_CALL yPlayer_SetSpawnInfo(yPlayer* player, yClassData const* data);
Y_API void Y_CALL yPlayer_GetClassData(yPlayer* player, yClassData* data);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
