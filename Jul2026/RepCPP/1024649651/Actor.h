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
    void (*OnPlayerGiveDamageActor)(yPlayer* player, yActor* actor, float amount, unsigned weapon, yBodyPart part);
    void (*OnActorStreamOut)(yActor* actor, yPlayer* forPlayer);
    void (*OnActorStreamIn)(yActor* actor, yPlayer* forPlayer);
} yActorEventHandler;

Y_API void Y_CALL yAddActorEventHandler(yActorEventHandler const* handler);

/// Create an actor
Y_API yActor* Y_CALL yActor_Create(int skin, yVector3 pos, float angle);

/// Destroy an actor
Y_API void Y_CALL yActor_Destroy(yActor* actor);

/// Get the actor's position
Y_API yVector3 Y_CALL yActor_GetPosition(yActor const* actor);

/// Set the actor's position
Y_API void Y_CALL yActor_SetPosition(yActor* actor, yVector3 position);

/// Get the actor's rotation
Y_API yQuat Y_CALL yActor_GetRotation(yActor const* actor);

/// Set the actor's rotation
Y_API void Y_CALL yActor_SetRotation(yActor* actor, yQuat rotation);

/// Get the actor's virtual world
Y_API int Y_CALL yActor_GetVirtualWorld(yActor const* actor);

/// Set the actor's virtual world
Y_API void Y_CALL yActor_SetVirtualWorld(yActor* actor, int vw);

/// Sets the actor's skin
Y_API void Y_CALL yActor_SetSkin(yActor* actor, int id);

/// Gets the actor's model
Y_API int Y_CALL yActor_GetSkin(yActor const* actor);

/// Apply an animation for the actor
Y_API void Y_CALL yActor_ApplyAnimation(yActor* actor, yAnimationData const* animation);

/// Clear the actor's animations
Y_API void Y_CALL yActor_ClearAnimations(yActor* actor);

/// Set the actor's health
Y_API void Y_CALL yActor_SetHealth(yActor* actor, float health);

/// Get the actor's health
Y_API float Y_CALL yActor_GetHealth(yActor const* actor);

/// Set whether the actor is invulnerable
Y_API void Y_CALL yActor_SetInvulnerable(yActor* actor, bool invuln);

/// Get whether the actor is invulnerable
Y_API bool Y_CALL yActor_IsInvulnerable(yActor const* actor);

/// Checks if actor is streamed for a player
Y_API bool Y_CALL yActor_IsStreamedInForPlayer(yActor const* actor, yPlayer const* player);

/// Streams actor for a player
Y_API void Y_CALL yActor_StreamInForPlayer(yActor* actor, yPlayer* player);

/// Streams out actor for a player
Y_API void Y_CALL yActor_StreamOutForPlayer(yActor* actor, yPlayer* player);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
