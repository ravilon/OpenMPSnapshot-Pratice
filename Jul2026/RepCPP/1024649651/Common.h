/*-------------------------------------------------------------------
*  Copyright (c) 2025 Maicol Castro <maicolcastro.abc@gmail.com>.
*  All rights reserved.
*
*  Distributed under the BSD 3-Clause License.
*  See LICENSE.txt in the root directory of this project or at
*  https://opensource.org/license/bsd-3-clause.
*-----------------------------------------------------------------*/

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "Export.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#ifdef Y_COMPONENT
#define Y_API Y_EXPORT

typedef struct IActor yActor;
typedef struct IBaseGangZone yGangZoneBase;
typedef struct IBasePickup yPickupBase;
typedef struct IClass yClass;
typedef struct IObject yObject;
typedef struct IPlayer yPlayer;
typedef struct IPlayerObject yPlayerObject;
typedef struct IPlayerTextDraw yPlayerTextDraw;
typedef struct ITextDraw yTextDraw;
typedef struct IVehicle yVehicle;
typedef struct ITextLabel yTextLabel;
typedef struct IPlayerTextLabel yPlayerTextLabel;
#else
#define Y_API Y_IMPORT

typedef struct {} yActor;
typedef struct {} yClass;
typedef struct {} yGangZoneBase;
typedef struct {} yObject;
typedef struct {} yPickupBase;
typedef struct {} yPlayer;
typedef struct {} yPlayerObject;
typedef struct {} yPlayerTextDraw;
typedef struct {} yTextDraw;
typedef struct {} yVehicle;
typedef struct {} yTextLabel;
typedef struct {} yPlayerTextLabel;
#endif

#define Y_MAX_WEAPON_SLOTS 13

typedef struct {
char const* chars;
size_t length;
} yStringView;

typedef struct {
float x;
float y;
} yVector2;

typedef struct {
float x;
float y;
float z;
} yVector3;

typedef struct {
float w;
float x;
float y;
float z;
} yVector4;

typedef yVector4 yQuat;

typedef struct {
float delta;
bool loop;
bool lockX;
bool lockY;
bool freeze;
uint32_t time;
yStringView library;
yStringView name;
} yAnimationData;

typedef enum {
Y_BODY_PART_TORSO = 3,
Y_BODY_PART_GROIN,
Y_BODY_PART_LEFT_ARM,
Y_BODY_PART_RIGHT_ARM,
Y_BODY_PART_LEFT_LEG,
Y_BODY_PART_RIGHT_LEG,
Y_BODY_PART_HEAD
} yBodyPart;

typedef struct {
uint8_t id;
uint32_t ammo;
} yWeaponSlotData;

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
