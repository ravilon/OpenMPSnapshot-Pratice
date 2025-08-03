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
Y_OBJECT_MATERIAL_SIZE_32X32 = 10,
Y_OBJECT_MATERIAL_SIZE_64X32 = 20,
Y_OBJECT_MATERIAL_SIZE_64X64 = 30,
Y_OBJECT_MATERIAL_SIZE_128X32 = 40,
Y_OBJECT_MATERIAL_SIZE_128X64 = 50,
Y_OBJECT_MATERIAL_SIZE_128X128 = 60,
Y_OBJECT_MATERIAL_SIZE_256X32 = 70,
Y_OBJECT_MATERIAL_SIZE_256X64 = 80,
Y_OBJECT_MATERIAL_SIZE_256X128 = 90,
Y_OBJECT_MATERIAL_SIZE_256X256 = 100,
Y_OBJECT_MATERIAL_SIZE_512X64 = 110,
Y_OBJECT_MATERIAL_SIZE_512X128 = 120,
Y_OBJECT_MATERIAL_SIZE_512X256 = 130,
Y_OBJECT_MATERIAL_SIZE_512X512 = 140
} yObjectMaterialSize;

typedef enum {
Y_OBJECT_MATERIAL_TEXT_ALIGN_LEFT,
Y_OBJECT_MATERIAL_TEXT_ALIGN_CENTER,
Y_OBJECT_MATERIAL_TEXT_ALIGN_RIGHT
} yObjectMaterialTextAlign;

typedef enum {
Y_OBJECT_EDIT_RESPONSE_CANCEL,
Y_OBJECT_EDIT_RESPONSE_FINAL,
Y_OBJECT_EDIT_RESPONSE_UPDATE
} yObjectEditResponse;

typedef struct {
int model;
int bone;
yVector3 offset;
yVector3 rotation;
yVector3 scale;
uint32_t colour1;
uint32_t colour2;
} yObjectAttachmentSlotData;

typedef struct {
void (Y_CALL* OnMoved)(yObject* object);
void (Y_CALL* OnPlayerObjectMoved)(yPlayer* player, yPlayerObject* object);
void (Y_CALL* OnObjectSelected)(yPlayer* player, yObject* object, int model, yVector3 position);
void (Y_CALL* OnPlayerObjectSelected)(yPlayer* player, yPlayerObject* object, int model, yVector3 position);
void (Y_CALL* OnObjectEdited)(yPlayer* player, yObject* object, yObjectEditResponse response, yVector3 offset, yVector3 rotation);
void (Y_CALL* OnPlayerObjectEdited)(yPlayer* player, yPlayerObject* object, yObjectEditResponse response, yVector3 offset, yVector3 rotation);
void (Y_CALL* OnPlayerAttachedObjectEdited)(yPlayer* player, int index, bool saved, yObjectAttachmentSlotData const* data);
} yObjectEventHandler;

Y_API void Y_CALL yAddObjectEventHandler(yObjectEventHandler const* handler);

/// Set the default camera collision for new objects
Y_API void Y_CALL ySetDefaultObjectCameraCollision(bool collision);

/// Get the default camera collision for new objects
Y_API bool Y_CALL yGetDefaultObjectCameraCollision();

/// Create a new object
/// @return A pointer if succeeded or NULL on failure
Y_API yObject* yObject_Create(int modelID, yVector3 position, yVector3 rotation, float drawDist);

Y_API void Y_CALL yObject_Destroy(yObject* object);

/// Get the object's position
Y_API yVector3 Y_CALL yObject_GetPosition(yObject const* object);

/// Set the object's position
Y_API void Y_CALL yObject_SetPosition(yObject* object, yVector3 position);

/// Get the object's rotation
Y_API yQuat Y_CALL yObject_GetRotation(yObject const* object);

/// Set the object's rotation
Y_API void Y_CALL yObject_SetRotation(yObject* object, yQuat rotation);

/// Get the object's virtual world
Y_API int Y_CALL yObject_GetVirtualWorld(yObject const* object);

/// Set the object's virtual world
Y_API void Y_CALL yObject_SetVirtualWorld(yObject* object, int vw);

/// Set the draw distance of the object
Y_API void Y_CALL yObject_SetDrawDistance(yObject* object, float drawDistance);

/// Get the object's draw distance
Y_API float Y_CALL yObject_GetDrawDistance(yObject const* object);

/// Set the model of the object
Y_API void Y_CALL yObject_SetModel(yObject* object, int model);

/// Get the object's model
Y_API int Y_CALL yObject_GetModel(yObject const* object);

/// Set whether the object has camera collision
Y_API void Y_CALL yObject_SetCameraCollision(yObject* object, bool collision);

/// Get whether the object has camera collision
Y_API bool Y_CALL yObject_GetCameraCollision(yObject const* object);

/// Start moving the object
Y_API void Y_CALL yObject_Move(yObject* object, yVector3 targetPos, yVector3 targetRot, float speed);

/// Get whether the object is moving
Y_API bool Y_CALL yObject_IsMoving(yObject const* object);

/// Stop moving the object prematurely
Y_API void Y_CALL yObject_Stop(yObject* object);

/// Attach the object to a vehicle
Y_API void Y_CALL yObject_AttachToVehicle(yObject* object, yVehicle* vehicle, yVector3 offset, yVector3 rotation);

/// Reset any attachment data about the object
Y_API void Y_CALL yObject_ResetAttachment(yObject* object);

/// Set the object's material to a texture
Y_API void Y_CALL yObject_SetMaterial(yObject* object, uint32_t materialIndex, int model, yStringView textureLibrary, yStringView textureName, uint32_t colour);

/// Set the object's material to some text
Y_API void Y_CALL yObject_SetMaterialText(yObject* object, uint32_t materialIndex, yStringView text, yObjectMaterialSize materialSize, yStringView fontFace, int fontSize, bool bold, uint32_t fontColour, uint32_t backgroundColour, yObjectMaterialTextAlign align);

/// Attach the object to a player
Y_API void Y_CALL yObject_AttachToPlayer(yObject* object, yPlayer* player, yVector3 offset, yVector3 rotation);

/// Attach the object to another object
Y_API void Y_CALL yObject_AttachToObject(yObject* object, yObject* target, yVector3 offset, yVector3 rotation, bool syncRotation);

/* ---------------------------------------------------------------- */

/// Create a new player object
/// @return A pointer if succeeded or NULL on failure
Y_API yPlayerObject* Y_CALL yPlayerObject_Create(yPlayer* player, int modelID, yVector3 position, yVector3 rotation);

Y_API void Y_CALL yPlayerObject_Destroy(yPlayer* player, yPlayerObject* object);

/// Get the playerobject's position
Y_API yVector3 Y_CALL yPlayerObject_GetPosition(yPlayerObject const* playerobject);

/// Set the playerobject's position
Y_API void Y_CALL yPlayerObject_SetPosition(yPlayerObject* playerobject, yVector3 position);

/// Get the playerobject's rotation
Y_API yQuat Y_CALL yPlayerObject_GetRotation(yPlayerObject const* playerobject);

/// Set the playerobject's rotation
Y_API void Y_CALL yPlayerObject_SetRotation(yPlayerObject* playerobject, yQuat rotation);

/// Get the playerobject's virtual world
Y_API int Y_CALL yPlayerObject_GetVirtualWorld(yPlayerObject const* playerObject);

/// Set the playerobject's virtual world
Y_API void Y_CALL yPlayerObject_SetVirtualWorld(yPlayerObject* playerobject, int vw);

/// Set the draw distance of the object
Y_API void Y_CALL yPlayerObject_SetDrawDistance(yPlayerObject* object, float drawDistance);

/// Get the object's draw distance
Y_API float Y_CALL yPlayerObject_GetDrawDistance(yPlayerObject const* object);

/// Set the model of the object
Y_API void Y_CALL yPlayerObject_SetModel(yPlayerObject* object, int model);

/// Get the object's model
Y_API int Y_CALL yPlayerObject_GetModel(yPlayerObject const* object);

/// Set whether the object has camera collision
Y_API void Y_CALL yPlayerObject_SetCameraCollision(yPlayerObject* object, bool collision);

/// Get whether the object has camera collision
Y_API bool Y_CALL yPlayerObject_GetCameraCollision(yPlayerObject const* object);

/// Start moving the object
Y_API void Y_CALL yPlayerObject_Move(yPlayerObject* object, yVector3 targetPos, yVector3 targetRot, float speed);

/// Get whether the object is moving
Y_API bool Y_CALL yPlayerObject_IsMoving(yPlayerObject const* object);

/// Stop moving the object prematurely
Y_API void Y_CALL yPlayerObject_Stop(yPlayerObject* object);

/// Attach the object to a vehicle
Y_API void Y_CALL yPlayerObject_AttachToVehicle(yPlayerObject* object, yVehicle* vehicle, yVector3 offset, yVector3 rotation);

/// Reset any attachment data about the object
Y_API void Y_CALL yPlayerObject_ResetAttachment(yPlayerObject* object);

/// Set the object's material to a texture
Y_API void Y_CALL yPlayerObject_SetMaterial(yPlayerObject* object, uint32_t materialIndex, int model, yStringView textureLibrary, yStringView textureName, uint32_t colour);

/// Set the object's material to some text
Y_API void Y_CALL yPlayerObject_SetMaterialText(yPlayerObject* object, uint32_t materialIndex, yStringView text, yObjectMaterialSize materialSize, yStringView fontFace, int fontSize, bool bold, uint32_t fontColour, uint32_t backgroundColour, yObjectMaterialTextAlign align);

/// Attach the object to a player
Y_API void Y_CALL yPlayerObject_AttachToPlayer(yPlayerObject* object, yPlayer* player, yVector3 offset, yVector3 rotation);

/// Attach the object to another object
Y_API void Y_CALL yPlayerObject_AttachToObject(yPlayerObject* object, yPlayerObject* target, yVector3 offset, yVector3 rotation);

/* ---------------------------------------------------------------- */

/// Set the player's attached object in an attachment slot
Y_API void Y_CALL yPlayer_SetAttachedObject(yPlayer* player, int index, yObjectAttachmentSlotData const* data);

/// Remove the player's attached object in an attachment slot
Y_API void Y_CALL yPlayer_RemoveAttachedObject(yPlayer* player, int index);

/// Check if the player has an attached object in an attachment slot
Y_API bool Y_CALL yPlayer_HasAttachedObject(yPlayer* player, int index);

/// Get the player's attached object in an attachment slot
Y_API void Y_CALL yPlayer_GetAttachedObject(yPlayer* player, int index, yObjectAttachmentSlotData* data);

/// Initiate object selection for the player
Y_API void Y_CALL yPlayer_BeginObjectSelecting(yPlayer* player);

/// Get whether the player is selecting objects
Y_API bool Y_CALL yPlayer_IsSelectingObject(yPlayer* player);

/// End selection and editing objects for the player
Y_API void Y_CALL yPlayer_EndObjectEditing(yPlayer* player);

/// Edit the object for the player
Y_API void Y_CALL yPlayer_BeginEditingObject(yPlayer* player, yObject* object);

/// Edit the player object for the player
Y_API void Y_CALL yPlayer_BeginEditingPlayerObject(yPlayer* player, yPlayerObject* object);

/// Check if the player is editing an object
Y_API bool Y_CALL yPlayer_IsEditingObject(yPlayer* player);

/// Edit an attached object in an attachment slot for the player
Y_API void Y_CALL yPlayer_EditAttachedObject(yPlayer* player, int index);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
