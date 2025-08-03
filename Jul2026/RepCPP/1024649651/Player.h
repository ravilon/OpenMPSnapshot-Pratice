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

#define Y_MAX_PLAYER_NAME_LEN 24

typedef enum {
Y_PLAYER_DISCONNECT_REASON_TIMEOUT,
Y_PLAYER_DISCONNECT_REASON_QUIT,
Y_PLAYER_DISCONNECT_REASON_KICKED,
} yPlayerDisconnectReason;

typedef enum {
Y_CLIENT_VERSION_SAMP_037,
Y_CLIENT_VERSION_SAMP_03DL,
Y_CLIENT_VERSION_OPENMP
} yClientVersion;

typedef enum {
Y_PLAYER_CAMERA_CUT_TYPE_CUT,
Y_PLAYER_CAMERA_CUT_TYPE_MOVE
} yPlayerCameraCutType;

typedef enum {
Y_MAP_ICON_STYLE_LOCAL,
Y_MAP_ICON_STYLE_GLOBAL,
Y_MAP_ICON_STYLE_LOCAL_CHECKPOINT,
Y_MAP_ICON_STYLE_GLOBAL_CHECKPOINT
} yMapIconStyle;

typedef enum {
Y_PLAYER_ANIMATION_SYNC_TYPE_NO_SYNC,
Y_PLAYER_ANIMATION_SYNC_TYPE_SYNC,
Y_PLAYER_ANIMATION_SYNC_TYPE_SYNC_OTHERS
} yPlayerAnimationSyncType;

typedef enum {
Y_PLAYER_STATE_NONE = 0,
Y_PLAYER_STATE_ON_FOOT = 1,
Y_PLAYER_STATE_DRIVER = 2,
Y_PLAYER_STATE_PASSENGER = 3,
Y_PLAYER_STATE_EXIT_VEHICLE = 4,
Y_PLAYER_STATE_ENTER_VEHICLE_DRIVER = 5,
Y_PLAYER_STATE_ENTER_VEHICLE_PASSENGER = 6,
Y_PLAYER_STATE_WASTED = 7,
Y_PLAYER_STATE_SPAWNED = 8,
Y_PLAYER_STATE_SPECTATING = 9
} yPlayerState;

typedef enum {
Y_PLAYER_FIGHTING_STYLE_NORMAL = 4,
Y_PLAYER_FIGHTING_STYLE_BOXING = 5,
Y_PLAYER_FIGHTING_STYLE_KUNG_FU = 6,
Y_PLAYER_FIGHTING_STYLE_KNEE_HEAD = 7,
Y_PLAYER_FIGHTING_STYLE_GRAB_KICK = 15,
Y_PLAYER_FIGHTING_STYLE_ELBOW = 16
} yPlayerFightingStyle;

typedef enum {
Y_PLAYER_WEAPON_SKILL_PISTOL,
Y_PLAYER_WEAPON_SKILL_SILENCED_PISTOL,
Y_PLAYER_WEAPON_SKILL_DESERT_EAGLE,
Y_PLAYER_WEAPON_SKILL_SHOTGUN,
Y_PLAYER_WEAPON_SKILL_SAWN_OFF,
Y_PLAYER_WEAPON_SKILL_SPAS12,
Y_PLAYER_WEAPON_SKILL_UZI,
Y_PLAYER_WEAPON_SKILL_MP5,
Y_PLAYER_WEAPON_SKILL_AK47,
Y_PLAYER_WEAPON_SKILL_M4,
Y_PLAYER_WEAPON_SKILL_SNIPER
} yPlayerWeaponSkill;

typedef enum {
Y_SPECIAL_ACTION_NONE,
Y_SPECIAL_ACTION_DUCK,
Y_SPECIAL_ACTION_JETPACK,
Y_SPECIAL_ACTION_ENTER_VEHICLE,
Y_SPECIAL_ACTION_EXIT_VEHICLE,
Y_SPECIAL_ACTION_DANCE1,
Y_SPECIAL_ACTION_DANCE2,
Y_SPECIAL_ACTION_DANCE3,
Y_SPECIAL_ACTION_DANCE4,
Y_SPECIAL_ACTION_HANDS_UP = 10,
Y_SPECIAL_ACTION_CELLPHONE,
Y_SPECIAL_ACTION_SITTING,
Y_SPECIAL_ACTION_STOP_CELLPHONE,
Y_SPECIAL_ACTION_BEER = 20,
Y_SPECIAL_ACTION_SMOKE,
Y_SPECIAL_ACTION_WINE,
Y_SPECIAL_ACTION_SPRUNK,
Y_SPECIAL_ACTION_CUFFED,
Y_SPECIAL_ACTION_CARRY,
Y_SPECIAL_ACTION_PISSING = 68
} yPlayerSpecialAction;

typedef enum {
Y_PLAYER_WEAPON_STATE_UNKNOWN = -1,
Y_PLAYER_WEAPON_STATE_NO_BULLETS,
Y_PLAYER_WEAPON_STATE_LAST_BULLET,
Y_PLAYER_WEAPON_STATE_MORE_BULLETS,
Y_PLAYER_WEAPON_STATE_RELOADING
} yPlayerWeaponState;

typedef enum {
Y_PLAYER_BULLET_HIT_TYPE_NONE,
Y_PLAYER_BULLET_HIT_TYPE_PLAYER = 1,
Y_PLAYER_BULLET_HIT_TYPE_VEHICLE = 2,
Y_PLAYER_BULLET_HIT_TYPE_OBJECT = 3,
Y_PLAYER_BULLET_HIT_TYPE_PLAYER_OBJECT = 4,
} yPlayerBulletHitType;

typedef enum {
Y_PLAYER_SPECTATE_MODE_NORMAL = 1,
Y_PLAYER_SPECTATE_MODE_FIXED,
Y_PLAYER_SPECTATE_MODE_SIDE
} yPlayerSpectateMode;

typedef enum {
Y_PLAYER_SPECTATE_TYPE_NONE,
Y_PLAYER_SPECTATE_TYPE_VEHICLE,
Y_PLAYER_SPECTATE_TYPE_PLAYER,
} yPlayerSpectateType;

typedef struct {
yVector3 camFrontVector;
yVector3 camPos;
float aimZ;
float camZoom;
float aspectRatio;
yPlayerWeaponState weaponState;
uint8_t camMode;
} yPlayerAimData;

typedef struct {
yVector3 origin;
yVector3 hitPos;
yVector3 offset;
uint8_t weapon;
yPlayerBulletHitType hitType;
uint16_t hitId;
} yPlayerBulletData;

typedef struct {
bool spectating;
int spectateId;
yPlayerSpectateType type;
} yPlayerSpectateData;

typedef struct {
bool (Y_CALL* OnPlayerRequestSpawn)(yPlayer* player);
void (Y_CALL* OnPlayerSpawn)(yPlayer* player);
} yPlayerSpawnEventHandler;

typedef struct {
void (Y_CALL* OnIncomingConnection)(yPlayer* player, yStringView ipAddress, unsigned short port);
void (Y_CALL* OnPlayerConnect)(yPlayer* player);
void (Y_CALL* OnPlayerDisconnect)(yPlayer* player, yPlayerDisconnectReason reason);
void (Y_CALL* OnPlayerClientInit)(yPlayer* player);
} yPlayerConnectEventHandler;

typedef struct {
void (Y_CALL* OnPlayerStreamIn)(yPlayer* player, yPlayer* forPlayer);
void (Y_CALL* OnPlayerStreamOut)(yPlayer* player, yPlayer* forPlayer);
} yPlayerStreamEventHandler;

typedef struct {
bool (Y_CALL* OnPlayerText)(yPlayer* player, yStringView message);
bool (Y_CALL* OnPlayerCommandText)(yPlayer* player, yStringView message);
} yPlayerTextEventHandler;

typedef struct {
bool (Y_CALL* OnPlayerShotMissed)(yPlayer* player, yPlayerBulletData const* bulletData);
bool (Y_CALL* OnPlayerShotPlayer)(yPlayer* player, yPlayer* target, yPlayerBulletData const* bulletData);
bool (Y_CALL* OnPlayerShotVehicle)(yPlayer* player, yVehicle* target, yPlayerBulletData const* bulletData);
bool (Y_CALL* OnPlayerShotObject)(yPlayer* player, yObject* target, yPlayerBulletData const* bulletData);
bool (Y_CALL* OnPlayerShotPlayerObject)(yPlayer* player, yPlayerObject* target, yPlayerBulletData const* bulletData);
} yPlayerShotEventHandler;

typedef struct {
void (Y_CALL* OnPlayerScoreChange)(yPlayer* player, int score);
void (Y_CALL* OnPlayerNameChange)(yPlayer* player, yStringView oldName);
void (Y_CALL* OnPlayerInteriorChange)(yPlayer* player, unsigned newInterior, unsigned oldInterior);
void (Y_CALL* OnPlayerStateChange)(yPlayer* player, yPlayerState newState, yPlayerState oldState);
void (Y_CALL* OnPlayerKeyStateChange)(yPlayer* player, uint32_t newKeys, uint32_t oldKeys);
} yPlayerChangeEventHandler;

typedef struct {
void (Y_CALL* OnPlayerDeath)(yPlayer* player, yPlayer* killer, int reason);
void (Y_CALL* OnPlayerTakeDamage)(yPlayer* player, yPlayer* from, float amount, unsigned weapon, yBodyPart part);
void (Y_CALL* OnPlayerGiveDamage)(yPlayer* player, yPlayer* to, float amount, unsigned weapon, yBodyPart part);
} yPlayerDamageEventHandler;

typedef struct {
void (Y_CALL* OnPlayerClickMap)(yPlayer* player, yVector3 pos);
void (Y_CALL* OnPlayerClickPlayer)(yPlayer* player, yPlayer* clicked);
} yPlayerClickEventHandler;

typedef struct {
void (Y_CALL* OnClientCheckResponse)(yPlayer* player, int actionType, int address, int results);
} yPlayerCheckEventHandler;

typedef struct {
bool (Y_CALL* OnPlayerUpdate)(yPlayer* player);
} yPlayerUpdateEventHandler;

Y_API void Y_CALL yAddPlayerSpawnEventHandler(yPlayerSpawnEventHandler const* handler);
Y_API void Y_CALL yAddPlayerConnectEventHandler(yPlayerConnectEventHandler const* handler);
Y_API void Y_CALL yAddPlayerStreamEventHandler(yPlayerStreamEventHandler const* handler);
Y_API void Y_CALL yAddPlayerTextEventHandler(yPlayerTextEventHandler const* handler);
Y_API void Y_CALL yAddPlayerShotEventHandler(yPlayerShotEventHandler const* handler);
Y_API void Y_CALL yAddPlayerChangeEventHandler(yPlayerChangeEventHandler const* handler);
Y_API void Y_CALL yAddPlayerDamageEventHandler(yPlayerDamageEventHandler const* handler);
Y_API void Y_CALL yAddPlayerClickEventHandler(yPlayerClickEventHandler const* handler);
Y_API void Y_CALL yAddPlayerCheckEventHandler(yPlayerCheckEventHandler const* handler);
Y_API void Y_CALL yAddPlayerUpdateEventHandler(yPlayerUpdateEventHandler const* handler);

/// SendClientMessage for all players
Y_API void Y_CALL yPlayer_SendClientMessageToAll(uint32_t colour, yStringView message);

/// SendChatMessage for all players
Y_API void Y_CALL yPlayer_SendChatMessageToAll(yPlayer* from, yStringView message);

/// SendGameText for all players
Y_API void Y_CALL yPlayer_SendGameTextToAll(yStringView message, uint32_t milliseconds, int style);

/// HideGameText for all players
Y_API void Y_CALL yPlayer_HideGameTextForAll(int style);

/// SendDeathMessage for all players
Y_API void Y_CALL yPlayer_SendDeathMessageToAll(yPlayer* killer, yPlayer* killee, int weapon);

/// CreateExplosion for all players
Y_API void Y_CALL yPlayer_CreateExplosionForAll(yVector3 vec, int type, float radius);

/// Allow or disallow the use of specific character in player names.
Y_API void Y_CALL yPlayer_AllowNickNameCharacter(char character, bool allow);

/// Check if a specific character is allowed to be used in player names.
Y_API bool Y_CALL yPlayer_IsNickNameCharacterAllowed(char character);

/// Get the player's position
Y_API yVector3 Y_CALL yPlayer_GetPosition(yPlayer const* player);

/// Set the player's position
Y_API void Y_CALL yPlayer_SetPosition(yPlayer* player, yVector3 position);

/// Get the player's rotation
Y_API yQuat Y_CALL yPlayer_GetRotation(yPlayer const* player);

/// Set the player's rotation
Y_API void Y_CALL yPlayer_SetRotation(yPlayer* player, yQuat rotation);

/// Get the player's virtual world
Y_API int Y_CALL yPlayer_GetVirtualWorld(yPlayer const* player);

/// Set the player's virtual world
Y_API void Y_CALL yPlayer_SetVirtualWorld(yPlayer* player, int vw);

/// Kick the player
Y_API void Y_CALL yPlayer_Kick(yPlayer* player);

/// Ban the player
Y_API void Y_CALL yPlayer_Ban(yPlayer* player, yStringView reason);

/// Get whether the player is a bot (NPC)
Y_API bool Y_CALL yPlayer_IsBot(yPlayer const* player);

/// Get the peer's ping from their network
Y_API unsigned Y_CALL yPlayer_GetPing(yPlayer* player);

/// Immediately spawn the player
Y_API void Y_CALL yPlayer_Spawn(yPlayer* player);

/// Get the player's client version
Y_API yClientVersion Y_CALL yPlayer_GetClientVersion(yPlayer const* player);

/// Set the player's position with the proper Z coordinate for the map
Y_API void Y_CALL yPlayer_SetPositionFindZ(yPlayer* player, yVector3 pos);

/// Set the player's camera position
Y_API void Y_CALL yPlayer_SetCameraPosition(yPlayer* player, yVector3 pos);

/// Get the player's camera position
Y_API yVector3 Y_CALL yPlayer_GetCameraPosition(yPlayer* player);

/// Set the direction a player's camera looks at
Y_API void Y_CALL yPlayer_SetCameraLookAt(yPlayer* player, yVector3 pos, yPlayerCameraCutType cutType);

/// Get the direction a player's camera looks at
Y_API yVector3 Y_CALL yPlayer_GetCameraLookAt(yPlayer* player);

/// Sets the camera to a place behind the player
Y_API void Y_CALL yPlayer_SetCameraBehind(yPlayer* player);

/// Interpolate camera position
Y_API void Y_CALL yPlayer_InterpolateCameraPosition(yPlayer* player, yVector3 from, yVector3 to, int time, yPlayerCameraCutType cutType);

/// Interpolate camera look at
Y_API void Y_CALL yPlayer_InterpolateCameraLookAt(yPlayer* player, yVector3 from, yVector3 to, int time, yPlayerCameraCutType cutType);

/// Attach player's camera to an object
Y_API void Y_CALL yPlayer_AttachCameraToObject(yPlayer* player, yObject* object);

/// Attach player's camera to a player object
Y_API void Y_CALL yPlayer_AttachCameraToPlayerObject(yPlayer* player, yPlayerObject* object);

/// Set the player's name
/// @return The player's new name status
Y_API int Y_CALL yPlayer_SetName(yPlayer* player, yStringView name);

/// Get the player's name
Y_API yStringView Y_CALL yPlayer_GetName(yPlayer const* player);

/// Get the player's serial (gpci)
Y_API yStringView Y_CALL yPlayer_GetSerial(yPlayer const* player);

/// Give a weapon to the player
Y_API void Y_CALL yPlayer_GiveWeapon(yPlayer* player, uint8_t weapon, uint32_t ammo);

/// Removes player weapon
Y_API void Y_CALL yPlayer_RemoveWeapon(yPlayer* player, uint8_t weapon);

/// Set the player's ammo for a weapon
Y_API void Y_CALL yPlayer_SetWeaponAmmo(yPlayer* player, uint8_t weapon, uint32_t data);

/// Get player's weapons
Y_API void Y_CALL yPlayer_GetWeapons(yPlayer const* player, yWeaponSlotData weapons[Y_MAX_WEAPON_SLOTS]);

/// Get single weapon
Y_API yWeaponSlotData Y_CALL yPlayer_GetWeaponSlot(yPlayer* player, int slot);

/// Reset the player's weapons
Y_API void Y_CALL yPlayer_ResetWeapons(yPlayer* player);

/// Set the player's currently armed weapon
Y_API void Y_CALL yPlayer_SetArmedWeapon(yPlayer* player, uint32_t weapon);

/// Get the player's currently armed weapon
Y_API uint32_t Y_CALL yPlayer_GetArmedWeapon(yPlayer const* player);

/// Get the player's currently armed weapon ammo
Y_API uint32_t Y_CALL yPlayer_GetArmedWeaponAmmo(yPlayer const* player);

/// Set the player's shop name
Y_API void Y_CALL yPlayer_SetShopName(yPlayer* player, yStringView name);

/// Set the player's drunk level
Y_API void Y_CALL yPlayer_SetDrunkLevel(yPlayer* player, int level);

/// Get the player's drunk level
Y_API int Y_CALL yPlayer_GetDrunkLevel(yPlayer const* player);

/// Set the player's colour
Y_API void Y_CALL yPlayer_SetColour(yPlayer* player, uint32_t colour);

/// Get the player's colour
Y_API uint32_t Y_CALL yPlayer_GetColour(yPlayer const* player);

/// Set another player's colour for this player
Y_API void Y_CALL yPlayer_SetOtherColour(yPlayer* player, yPlayer* other, uint32_t colour);

/// Get another player's colour for this player
Y_API bool Y_CALL yPlayer_GetOtherColour(yPlayer const* player, yPlayer* other, uint32_t* colour);

/// Set whether the player is controllable
Y_API void Y_CALL yPlayer_SetControllable(yPlayer* player, bool controllable);

/// Get whether the player is controllable
Y_API bool Y_CALL yPlayer_GetControllable(yPlayer const* player);

/// Set whether the player is spectating
Y_API void Y_CALL yPlayer_SetSpectating(yPlayer* player, bool spectating);

/// Set the player's wanted level
Y_API void Y_CALL yPlayer_SetWantedLevel(yPlayer* player, unsigned level);

/// Get the player's wanted level
Y_API unsigned Y_CALL yPlayer_GetWantedLevel(yPlayer const* player);

/// Play a sound for the player at a position
/// @param sound The sound ID
/// @param pos The position to play at
Y_API void Y_CALL yPlayer_PlaySound(yPlayer* player, uint32_t sound, yVector3 pos);

/// Get the sound that was last played
Y_API uint32_t Y_CALL yPlayer_LastPlayedSound(yPlayer const* player);

/// Play an audio stream for the player
/// @param url The HTTP URL of the stream
/// @param[opt] usePos Whether to play in a radius at a specific position
/// @param pos The position to play at
/// @param distance The distance to play at
Y_API void Y_CALL yPlayer_PlayAudio(yPlayer* player, yStringView url, bool usePos, yVector3 pos, float distance);

Y_API bool Y_CALL yPlayer_PlayCrimeReport(yPlayer* player, yPlayer* suspect, uint32_t crime);

/// Stop playing audio stream for the player
Y_API void Y_CALL yPlayer_StopAudio(yPlayer* player);

/// Create an explosion
Y_API void Y_CALL yPlayer_CreateExplosion(yPlayer* player, yVector3 vec, int type, float radius);

/// Send Death message
Y_API void Y_CALL yPlayer_SendDeathMessage(yPlayer* player, yPlayer* victim, yPlayer* killer, int weapon);

/// Send empty death message
Y_API void Y_CALL yPlayer_SendEmptyDeathMessage(yPlayer* player);

/// Remove default map objects with a model in a radius at a specific position
/// @param model The object model to remove
/// @param pos The position to remove at
/// @param radius The radius to remove around
Y_API void Y_CALL yPlayer_RemoveDefaultObjects(yPlayer* player, unsigned model, yVector3 pos, float radius);

/// Force class selection for the player
Y_API void Y_CALL yPlayer_ForceClassSelection(yPlayer* player);

/// Set the player's money
Y_API void Y_CALL yPlayer_SetMoney(yPlayer* player, int money);

/// Give money to the player
Y_API void Y_CALL yPlayer_GiveMoney(yPlayer* player, int money);

/// Reset the player's money to 0
Y_API void Y_CALL yPlayer_ResetMoney(yPlayer* player);

/// Get the player's money
Y_API int Y_CALL yPlayer_GetMoney(yPlayer* player);

/// Set a map icon for the player
Y_API void Y_CALL yPlayer_SetMapIcon(yPlayer* player, int id, yVector3 pos, int type, uint32_t colour, yMapIconStyle style);

/// Unset a map icon for the player
Y_API void Y_CALL yPlayer_UnsetMapIcon(yPlayer* player, int id);

/// Toggle stunt bonus for the player
Y_API void Y_CALL yPlayer_UseStuntBonuses(yPlayer* player, bool enable);

/// Toggle another player's name tag for the player
Y_API void Y_CALL yPlayer_ToggleOtherNameTag(yPlayer* player, yPlayer* other, bool toggle);

/// Set the player's game time
/// @param hr The hours from 0 to 23
/// @param min The minutes from 0 to 59
Y_API void Y_CALL yPlayer_SetTime(yPlayer* player, int hr, int min);

/// Toggle the player's clock visibility
Y_API void Y_CALL yPlayer_UseClock(yPlayer* player, bool enable);

/// Get whether the clock is visible for the player
Y_API bool Y_CALL yPlayer_HasClock(yPlayer const* player);

/// Toggle widescreen for player
Y_API void Y_CALL yPlayer_UseWidescreen(yPlayer* player, bool enable);

/// Get widescreen status from player
Y_API bool Y_CALL yPlayer_HasWidescreen(yPlayer const* player);

/// Set the transform applied to player rotation
Y_API void Y_CALL yPlayer_SetTransform(yPlayer* player, yQuat tm);

/// Set the player's health
Y_API void Y_CALL yPlayer_SetHealth(yPlayer* player, float health);

/// Get the player's health
Y_API float Y_CALL yPlayer_GetHealth(yPlayer const* player);

/// Set the player's score
Y_API void Y_CALL yPlayer_SetScore(yPlayer* player, int score);

/// Get the player's score
Y_API int Y_CALL yPlayer_GetScore(yPlayer const* player);

/// Set the player's armour
Y_API void Y_CALL yPlayer_SetArmour(yPlayer* player, float armour);

/// Get the player's armour
Y_API float Y_CALL yPlayer_GetArmour(yPlayer const* player);

/// Set the player's gravity
Y_API void Y_CALL yPlayer_SetGravity(yPlayer* player, float gravity);

/// Get player's gravity
Y_API float Y_CALL yPlayer_GetGravity(yPlayer const* player);

/// Set the player's world time
Y_API void Y_CALL yPlayer_SetWorldTime(yPlayer* player, int time);

/// Apply an animation to the player
/// @param animation The animation to apply
/// @param syncType How to sync the animation
Y_API void Y_CALL yPlayer_ApplyAnimation(yPlayer* player, yAnimationData const* animation, yPlayerAnimationSyncType syncType);

/// Clear the player's animation
/// @param syncType How to sync the animation
Y_API void Y_CALL yPlayer_ClearAnimations(yPlayer* player, yPlayerAnimationSyncType syncType);

/// Get the player's animation data
Y_API void Y_CALL yPlayer_GetAnimationData(yPlayer const* player, uint16_t* id, uint16_t* flags);

// /// Get the player's surf data
// PlayerSurfingData yPlayer_GetSurfingData(yPlayer const* player);

/// Stream in the player for another player
/// @param other The player to stream in
Y_API void Y_CALL yPlayer_StreamInForPlayer(yPlayer* player, yPlayer* other);

/// Check if a player is streamed in for the current player
Y_API bool Y_CALL yPlayer_IsStreamedInForPlayer(yPlayer const* player, yPlayer const* other);

/// Stream out a player for the current player
/// @param other The player to stream out
Y_API void Y_CALL yPlayer_StreamOutForPlayer(yPlayer* player, yPlayer* other);

// /// Get the players which are streamed in for this player
// FlatPtrHashSet const<yPlayer>& yPlayer_StreamedForPlayers(yPlayer const* player);

/// Get the player's state
Y_API yPlayerState Y_CALL yPlayer_GetState(yPlayer const* player);

/// Set the player's team
Y_API void Y_CALL yPlayer_SetTeam(yPlayer* player, int team);

/// Get the player's team
Y_API int Y_CALL yPlayer_GetTeam(yPlayer const* player);

/// Set the player's skin
Y_API void Y_CALL yPlayer_SetSkin(yPlayer* player, int skin, bool send);

/// Get the player's skin
Y_API int Y_CALL yPlayer_GetSkin(yPlayer const* player);

Y_API void Y_CALL yPlayer_SetChatBubble(yPlayer* player, yStringView text, uint32_t colour, float drawDist, uint32_t milliseconds);

/// Send a message to the player
Y_API void Y_CALL yPlayer_SendClientMessage(yPlayer* player, uint32_t colour, yStringView message);

/// Send a standardly formatted chat message from another player
Y_API void Y_CALL yPlayer_SendChatMessage(yPlayer* player, yPlayer* sender, yStringView message);

/// Send a command to server (Player)
Y_API void Y_CALL yPlayer_SendCommand(yPlayer* player, yStringView message);

/// Send a game text message to the player
Y_API void Y_CALL yPlayer_SendGameText(yPlayer* player, yStringView message, uint32_t milliseconds, int style);

/// Hide a game text message from the player
Y_API void Y_CALL yPlayer_HideGameText(yPlayer* player, int style);

/// Check if the player can currently see this game text.
Y_API bool Y_CALL yPlayer_HasGameText(yPlayer* player, int style);

/// Set the player's weather
Y_API void Y_CALL yPlayer_SetWeather(yPlayer* player, int weatherID);

/// Get the player's weather
Y_API int Y_CALL yPlayer_GetWeather(yPlayer const* player);

/// Set world bounds
Y_API void Y_CALL yPlayer_SetWorldBounds(yPlayer* player, yVector4 coords);

/// Get world bounds
Y_API yVector4 Y_CALL yPlayer_GetWorldBounds(yPlayer const* player);

/// Set the player's fighting style
/// @note See https://open.mp/docs/scripting/resources/fightingstyles
Y_API void Y_CALL yPlayer_SetFightingStyle(yPlayer* player, yPlayerFightingStyle style);

/// Get the player's fighting style
/// @note See https://open.mp/docs/scripting/resources/fightingstyles
Y_API yPlayerFightingStyle Y_CALL yPlayer_GetFightingStyle(yPlayer const* player);

/// Set the player's skill level
/// @note See https://open.mp/docs/scripting/resources/weaponskills
/// @param skill The skill type
/// @param level The skill level
Y_API void Y_CALL yPlayer_SetSkillLevel(yPlayer* player, yPlayerWeaponSkill skill, int level);

/// Set the player's special action
Y_API void Y_CALL yPlayer_SetAction(yPlayer* player, yPlayerSpecialAction action);

/// Get the player's special action
Y_API yPlayerSpecialAction Y_CALL yPlayer_GetAction(yPlayer const* player);

/// Set the player's velocity
Y_API void Y_CALL yPlayer_SetVelocity(yPlayer* player, yVector3 velocity);

/// Get the player's velocity
Y_API yVector3 Y_CALL yPlayer_GetVelocity(yPlayer const* player);

/// Set the player's interior
Y_API void Y_CALL yPlayer_SetInterior(yPlayer* player, unsigned interior);

/// Get the player's interior
Y_API unsigned Y_CALL yPlayer_GetInterior(yPlayer const* player);

/// Get the player's key data
Y_API void Y_CALL yPlayer_GetKeyData(yPlayer const* player, uint32_t* keys, int16_t* upDown, int16_t* leftRight);

Y_API void Y_CALL yPlayer_GetSkillLevels(yPlayer const* player, uint16_t* skills);

/// Get the player's aim data
Y_API void Y_CALL yPlayer_GetAimData(yPlayer const* player, yPlayerAimData* data);

/// Get the player's bullet data
Y_API void Y_CALL yPlayer_GetBulletData(yPlayer const* player, yPlayerBulletData* data);

/// Toggle the camera targeting functions for the player
Y_API void Y_CALL yPlayer_UseCameraTargeting(yPlayer* player, bool enable);

/// Get whether the player has camera targeting functions enabled
Y_API bool Y_CALL yPlayer_HasCameraTargeting(yPlayer const* player);

/// Remove the player from their vehicle
Y_API void Y_CALL yPlayer_RemoveFromVehicle(yPlayer* player, bool force);

/// Get the player the player is looking at or nullptr if none
Y_API yPlayer* Y_CALL yPlayer_GetCameraTargetPlayer(yPlayer* player);

/// Get the vehicle the player is looking at or nullptr if none
Y_API yVehicle* Y_CALL yPlayer_GetCameraTargetVehicle(yPlayer* player);

/// Get the object the player is looking at or nullptr if none
Y_API yObject* Y_CALL yPlayer_GetCameraTargetObject(yPlayer* player);

/// Get the actor the player is looking at or nullptr if none
Y_API yActor* Y_CALL yPlayer_GetCameraTargetActor(yPlayer* player);

/// Get the player the player is targeting or nullptr if none
Y_API yPlayer* Y_CALL yPlayer_GetTargetPlayer(yPlayer* player);

/// Get the actor the player is targeting or nullptr if none
Y_API yActor* Y_CALL yPlayer_GetTargetActor(yPlayer* player);

/// Disable remote vehicle collision detection for this player.
Y_API void Y_CALL yPlayer_SetRemoteVehicleCollisions(yPlayer* player, bool collide);

/// Make player spectate another player
Y_API void Y_CALL yPlayer_SpectatePlayer(yPlayer* player, yPlayer* target, yPlayerSpectateMode mode);

/// Make player spectate a vehicle
Y_API void Y_CALL yPlayer_SpectateVehicle(yPlayer* player, yVehicle* target, yPlayerSpectateMode mode);

// /// Get spectate data
// yPlayerSpectateData const& yPlayer_GetSpectateData(yPlayer const* player);

/// Send client check (asks for certain data depending on type of action)
Y_API void Y_CALL yPlayer_SendClientCheck(yPlayer* player, int actionType, uint32_t address, int32_t offset, uint32_t count);

/// Toggle player's collision for other players
Y_API void Y_CALL yPlayer_ToggleGhostMode(yPlayer* player, bool toggle);

/// Get player's collision status (ghost mode)
Y_API bool Y_CALL yPlayer_IsGhostModeEnabled(yPlayer const* player);

/// Get default objects removed (basically just how many times removeDefaultObject is called)
Y_API int Y_CALL yPlayer_GetDefaultObjectsRemoved(yPlayer const* player);

/// Get if player is kicked or not (about to be disconnected)
Y_API bool Y_CALL yPlayer_GetKickStatus(yPlayer const* player);

/// Clear player tasks
Y_API void Y_CALL yPlayer_ClearTasks(yPlayer* player, yPlayerAnimationSyncType syncType);

/// Allow player to use weapons
Y_API void Y_CALL yPlayer_AllowWeapons(yPlayer* player, bool allow);

/// Check if player is allowed to use weapons
Y_API bool Y_CALL yPlayer_AreWeaponsAllowed(yPlayer const* player);

/// Teleport the player when they click the map
Y_API void Y_CALL yPlayer_AllowTeleport(yPlayer* player, bool allow);

/// Does the player teleport when they click the map
Y_API bool Y_CALL yPlayer_IsTeleportAllowed(yPlayer const* player);

/// Check if player is using an official client or not
Y_API bool Y_CALL yPlayer_IsUsingOfficialClient(yPlayer const* player);

/// Check if player is using omp or not
Y_API bool Y_CALL yPlayer_IsUsingOmp(yPlayer const* player);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
