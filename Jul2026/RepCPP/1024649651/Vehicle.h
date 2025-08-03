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
    Y_VEHICLE_SCM_EVENT_SET_PAINTJOB = 1,
    Y_VEHICLE_SCM_EVENT_ADD_COMPONENT,
    Y_VEHICLE_SCM_EVENT_SET_COLOUR,
    Y_VEHICLE_SCM_EVENT_ENTER_EXIT_MOD_SHOP
} yVehicleScmEvent;

typedef enum {
    Y_VEHICLE_COMPONENT_NONE = -1,
    Y_VEHICLE_COMPONENT_SPOILER = 0,
    Y_VEHICLE_COMPONENT_HOOD = 1,
    Y_VEHICLE_COMPONENT_ROOF = 2,
    Y_VEHICLE_COMPONENT_SIDE_SKIRT = 3,
    Y_VEHICLE_COMPONENT_LAMPS = 4,
    Y_VEHICLE_COMPONENT_NITRO = 5,
    Y_VEHICLE_COMPONENT_EXHAUST = 6,
    Y_VEHICLE_COMPONENT_WHEELS = 7,
    Y_VEHICLE_COMPONENT_STEREO = 8,
    Y_VEHICLE_COMPONENT_HYDRAULICS = 9,
    Y_VEHICLE_COMPONENT_FRONT_BUMPER = 10,
    Y_VEHICLE_COMPONENT_REAR_BUMPER = 11,
    Y_VEHICLE_COMPONENT_VENT_RIGHT = 12,
    Y_VEHICLE_COMPONENT_VENT_LEFT = 13,
    Y_VEHICLE_COMPONENT_FRONT_BULLBAR = 14,
    Y_VEHICLE_COMPONENT_REAR_BULLBAR = 15,
} yVehicleComponentSlot;

typedef enum {
    Y_VEHICLE_VELOCITY_SET_NORMAL = 0,
    Y_VEHICLE_VELOCITY_SET_ANGULAR
} yVehicleVelocitySetType;

typedef enum {
    Y_VEHICLE_MODEL_INFO_SIZE = 1,
    Y_VEHICLE_MODEL_INFO_FRONT_SEAT,
    Y_VEHICLE_MODEL_INFO_REAR_SEAT,
    Y_VEHICLE_MODEL_INFO_PETROL_CAP,
    Y_VEHICLE_MODEL_INFO_WHEELS_FRONT,
    Y_VEHICLE_MODEL_INFO_WHEELS_REAR,
    Y_VEHICLE_MODEL_INFO_WHEELS_MID,
    Y_VEHICLE_MODEL_INFO_FRONT_BUMPER_Z,
    Y_VEHICLE_MODEL_INFO_REAR_BUMPER_Z
} yVehicleModelInfoType;

typedef struct {
    uint32_t respawnDelaySeconds;
    int modelId;
    yVector3 position;
    float zRotation;
    int colour1;
    int colour2;
    bool siren;
    int interior;
} yVehicleSpawnData;

typedef struct {
    uint8_t seat;
    yVector3 position;
    yVector3 velocity;
} yUnoccupiedVehicleUpdate;

typedef struct {
    yVector3 size;
    yVector3 frontSeat;
    yVector3 rearSeat;
    yVector3 petrolCap;
    yVector3 frontWheel;
    yVector3 rearWheel;
    yVector3 midWheel;
    float frontBumperZ;
    float rearBumperZ;
} yVehicleModelInfo;

typedef struct {
    int8_t engine;
    int8_t lights;
    int8_t alarm;
    int8_t doors;
    int8_t bonnet;
    int8_t boot;
    int8_t objective;
    int8_t siren;
    int8_t doorDriver;
    int8_t doorPassenger;
    int8_t doorBackLeft;
    int8_t doorBackRight;
    int8_t windowDriver;
    int8_t windowPassenger;
    int8_t windowBackLeft;
    int8_t windowBackRight;
} yVehicleParams;

typedef struct {
    int playerId;
    uint16_t vehicleId;
    uint16_t leftRight;
    uint16_t upDown;
    uint16_t keys;
    yQuat rotation;
    yVector3 position;
    yVector3 velocity;
    float health;
    yVector2 playerHealthArmour;
    uint8_t siren;
    uint8_t landingGear;
    uint16_t trailerId;
    bool hasTrailer;

    union {
        uint8_t additionalKeyWeapon;
        struct {
            uint8_t weaponID : 6;
            uint8_t additionalKey : 2;
        };
    };

    union {
        uint32_t hydraThrustAngle;
        float trainSpeed;
    };
} yVehicleDriverSyncPacket;

typedef struct {
    int playerId;
    int vehicleId;

    union {
        uint16_t driveBySeatAdditionalKeyWeapon;

        struct {
            uint8_t seatID : 6;
            uint8_t driveBy : 1;
            uint8_t cuffed : 1;
            uint8_t weaponID : 6;
            uint8_t additionalKey : 2;
        };
    };

    uint16_t keys;
    yVector2 healthArmour;
    uint16_t leftRight;
    uint16_t upDown;
    yVector3 position;
} yVehiclePassengerSyncPacket;

typedef struct {
    int vehicleId;
    int playerId;
    uint8_t seatId;
    yVector3 roll;
    yVector3 rotation;
    yVector3 position;
    yVector3 velocity;
    yVector3 angularVelocity;
    float health;
} yVehicleUnoccupiedSyncPacket;

typedef struct {
    int vehicleId;
    int playerId;
    yVector3 position;
    yVector4 quat;
    yVector3 velocity;
    yVector3 turnVelocity;
} yVehicleTrailerSyncPacket;

typedef struct {
    void (Y_CALL* OnVehicleStreamIn)(yVehicle* vehicle, yPlayer* player);
    void (Y_CALL* OnVehicleStreamOut)(yVehicle* vehicle, yPlayer* player);
    void (Y_CALL* OnVehicleDeath)(yVehicle* vehicle, yPlayer* player);
    void (Y_CALL* OnPlayerEnterVehicle)(yPlayer* player, yVehicle* vehicle, bool passenger);
    void (Y_CALL* OnPlayerExitVehicle)(yPlayer* player, yVehicle* vehicle);
    void (Y_CALL* OnVehicleDamageStatusUpdate)(yVehicle* vehicle, yPlayer* player);
    bool (Y_CALL* OnVehiclePaintJob)(yPlayer* player, yVehicle* vehicle, int paintJob);
    bool (Y_CALL* OnVehicleMod)(yPlayer* player, yVehicle* vehicle, int component);
    bool (Y_CALL* OnVehicleRespray)(yPlayer* player, yVehicle* vehicle, int colour1, int colour2);
    void (Y_CALL* OnEnterExitModShop)(yPlayer* player, bool enterExit, int interiorId);
    void (Y_CALL* OnVehicleSpawn)(yVehicle* vehicle);
    bool (Y_CALL* OnUnoccupiedVehicleUpdate)(yVehicle* vehicle, yPlayer* player, yUnoccupiedVehicleUpdate const* updateData);
    bool (Y_CALL* OnTrailerUpdate)(yPlayer* player, yVehicle* trailer);
    bool (Y_CALL* OnVehicleSirenStateChange)(yPlayer* player, yVehicle* vehicle, uint8_t sirenState);
} yVehicleEventHandler;

Y_API void Y_CALL yAddVehicleEventHandler(yVehicleEventHandler const* handler);

Y_API bool Y_CALL yIsValidVehicleModel(int modelId);
Y_API uint8_t Y_CALL yGetVehiclePassengerSeats(int modelId);
Y_API bool Y_CALL yGetVehicleModelInfo(int modelId, yVehicleModelInfoType type, yVector3* out);
Y_API int Y_CALL yGetVehicleComponentSlot(int component);
Y_API bool Y_CALL yIsValidComponentForVehicleModel(int vehicleModel, int componentId);
Y_API void Y_CALL yGetRandomVehicleColour2(int modelId, int* colour1, int* colour2);
Y_API void Y_CALL yGetRandomVehicleColour4(int modelId, int* colour1, int* colour2, int* colour3, int* colour4);
Y_API uint32_t Y_CALL yVehicleColourIndexToRGBA(int index, uint32_t alpha);
Y_API bool Y_CALL yGetVehicleModelInfo(int modelId, yVehicleModelInfoType type, yVector3* out);

Y_API yVehicle* Y_CALL yVehicle_Create(bool isStatic, int modelID, yVector3 position, float rotZ, int colour1, int colour2, size_t respawnDelaySeconds, bool addSiren);
Y_API void Y_CALL yVehicle_Destroy(yVehicle* vehicle);

/// Set the inital spawn data of the vehicle
Y_API void Y_CALL yVehicle_SetSpawnData(yVehicle* vehicle, yVehicleSpawnData const* data);

/// Get the initial spawn data of the vehicle
Y_API void Y_CALL yVehicle_GetSpawnData(yVehicle* vehicle, yVehicleSpawnData* output);

/// Checks if player has the vehicle streamed in for themselves
Y_API bool Y_CALL yVehicle_IsStreamedInForPlayer(yVehicle const* vehicle, yPlayer const* player);

/// Streams in the vehicle for a specific player
Y_API void Y_CALL yVehicle_StreamInForPlayer(yVehicle* vehicle, yPlayer* player);

/// Streams out the vehicle for a specific player
Y_API void Y_CALL yVehicle_StreamOutForPlayer(yVehicle* vehicle, yPlayer* player);

/// Set the vehicle's colour
Y_API void Y_CALL yVehicle_SetColour(yVehicle* vehicle, int col1, int col2);

/// Get the vehicle's colour
Y_API void Y_CALL yVehicle_GetColour(yVehicle const* vehicle, int* colour1, int* colour2);

/// Set the vehicle's health
Y_API void Y_CALL yVehicle_SetHealth(yVehicle* vehicle, float health);

/// Get the vehicle's health
Y_API float Y_CALL yVehicle_GetHealth(yVehicle* vehicle);

/// Update the vehicle from a sync packet
Y_API bool Y_CALL yVehicle_UpdateFromDriverSync(yVehicle* vehicle, yVehicleDriverSyncPacket const* vehicleSync, yPlayer* player);

/// Update the vehicle from a passenger sync packet
Y_API bool Y_CALL yVehicle_UpdateFromPassengerSync(yVehicle* vehicle, yVehiclePassengerSyncPacket const* passengerSync, yPlayer* player);

/// Update the vehicle from an unoccupied sync packet
Y_API bool Y_CALL yVehicle_UpdateFromUnoccupied(yVehicle* vehicle, yVehicleUnoccupiedSyncPacket const* unoccupiedSync, yPlayer* player);

/// Update the vehicle from a trailer sync packet
Y_API bool Y_CALL yVehicle_UpdateFromTrailerSync(yVehicle* vehicle, yVehicleTrailerSyncPacket const* trailerSync, yPlayer* player);

/// Sets the vehicle's number plate
Y_API void Y_CALL yVehicle_SetPlate(yVehicle* vehicle, yStringView plate);

/// Get the vehicle's number plate
Y_API yStringView Y_CALL yVehicle_GetPlate(yVehicle* vehicle);

/// Sets the vehicle's damage status
Y_API void Y_CALL yVehicle_SetDamageStatus(yVehicle* vehicle, int panelStatus, int doorStatus, uint8_t lightStatus, uint8_t tyreStatus, yPlayer* vehicleUpdater);

/// Gets the vehicle's damage status
Y_API void Y_CALL yVehicle_GetDamageStatus(yVehicle* vehicle, int* panelStatus, int* doorStatus, int* lightStatus, int* tyreStatus);

/// Sets the vehicle's paintjob
Y_API void Y_CALL yVehicle_SetPaintJob(yVehicle* vehicle, int paintjob);

/// Gets the vehicle's paintjob
Y_API int Y_CALL yVehicle_GetPaintJob(yVehicle* vehicle);

/// Adds a component to the vehicle.
Y_API void Y_CALL yVehicle_AddComponent(yVehicle* vehicle, int component);

/// Gets the vehicle's component in a designated slot
Y_API int Y_CALL yVehicle_GetComponentInSlot(yVehicle* vehicle, int slot);

/// Removes a vehicle's component.
Y_API void Y_CALL yVehicle_RemoveComponent(yVehicle* vehicle, int component);

/// Puts the player inside this vehicle.
Y_API void Y_CALL yVehicle_PutPlayer(yVehicle* vehicle, yPlayer* player, int seatID);

/// Set the vehicle's Z angle.
Y_API void Y_CALL yVehicle_SetZAngle(yVehicle* vehicle, float angle);

/// Gets the vehicle's Z angle.
Y_API float Y_CALL yVehicle_GetZAngle(yVehicle* vehicle);

/// Set the vehicle's parameters.
Y_API void Y_CALL yVehicle_SetParams(yVehicle* vehicle, yVehicleParams const* params);

/// Set the vehicle's parameters for a specific player.
Y_API void Y_CALL yVehicle_SetParamsForPlayer(yVehicle* vehicle, yPlayer* player, yVehicleParams const* params);

/// Get the vehicle's parameters.
Y_API yVehicleParams const* Y_CALL yVehicle_GetParams(yVehicle* vehicle);

/// Checks if the vehicle is dead.
Y_API bool Y_CALL yVehicle_IsDead(yVehicle* vehicle);

/// Respawns the vehicle.
Y_API void Y_CALL yVehicle_Respawn(yVehicle* vehicle);

/// Get the vehicle's respawn delay in seconds.
Y_API size_t Y_CALL yVehicle_GetRespawnDelay(yVehicle* vehicle);

/// Set the vehicle respawn delay.
Y_API void Y_CALL yVehicle_SetRespawnDelay(yVehicle* vehicle, size_t seconds);

/// Checks if the vehicle is respawning.
Y_API bool Y_CALL yVehicle_IsRespawning(yVehicle* vehicle);

/// Sets (links) the vehicle to an interior.
Y_API void Y_CALL yVehicle_SetInterior(yVehicle* vehicle, int interiorID);

/// Gets the vehicle's interior.
Y_API int Y_CALL yVehicle_GetInterior(yVehicle* vehicle);

/// Attaches a vehicle as a trailer to this vehicle.
Y_API void Y_CALL yVehicle_AttachTrailer(yVehicle* vehicle, yVehicle* trailer);

/// Detaches a vehicle from this vehicle.
Y_API void Y_CALL yVehicle_DetachTrailer(yVehicle* vehicle);

/// Checks if the current vehicle is a trailer.
Y_API bool Y_CALL yVehicle_IsTrailer(yVehicle const* vehicle);

/// Get the current vehicle's attached trailer.
Y_API yVehicle* Y_CALL yVehicle_GetTrailer(yVehicle const* vehicle);

/// Get the current vehicle's cab.
Y_API yVehicle* Y_CALL yVehicle_GetCab(yVehicle const* vehicle);

/// Fully repair the vehicle.
Y_API void Y_CALL yVehicle_Repair(yVehicle* vehicle);

/// Adds a train carriage to the vehicle (ONLY FOR TRAINS).
Y_API void Y_CALL yVehicle_AddCarriage(yVehicle* vehicle, yVehicle* carriage, int pos);
Y_API void Y_CALL yVehicle_UpdateCarriage(yVehicle* vehicle, yVector3 pos, yVector3 veloc);

/// Sets the velocity of the vehicle.
Y_API void Y_CALL yVehicle_SetVelocity(yVehicle* vehicle, yVector3 velocity);

/// Gets the current velocity of the vehicle.
Y_API yVector3 Y_CALL yVehicle_GetVelocity(yVehicle* vehicle);

/// Sets the angular velocity of the vehicle.
Y_API void Y_CALL yVehicle_SetAngularVelocity(yVehicle* vehicle, yVector3 velocity);

/// Gets the current angular velocity of the vehicle.
Y_API yVector3 Y_CALL yVehicle_GetAngularVelocity(yVehicle* vehicle);

/// Gets the current model ID of the vehicle.
Y_API int Y_CALL yVehicle_GetModel(yVehicle* vehicle);

/// Gets the current landing gear state from a ID_VEHICLE_SYNC packet from the latest driver.
Y_API uint8_t Y_CALL yVehicle_GetLandingGearState(yVehicle* vehicle);

/// Get if the vehicle was occupied since last spawn.
Y_API bool Y_CALL yVehicle_HasBeenOccupied(yVehicle* vehicle);

/// Get the last time the vehicle was occupied.
Y_API size_t Y_CALL yVehicle_GetLastOccupiedTime(yVehicle* vehicle);

/// Get the last time the vehicle was spawned.
Y_API size_t Y_CALL yVehicle_GetLastSpawnTime(yVehicle* vehicle);

/// Get if vehicle is occupied.
Y_API bool Y_CALL yVehicle_IsOccupied(yVehicle* vehicle);

/// Set vehicle siren status.
Y_API void Y_CALL yVehicle_SetSiren(yVehicle* vehicle, bool status);

/// Get vehicle siren status.
Y_API uint8_t Y_CALL yVehicle_GetSirenState(yVehicle const* vehicle);

/// Get hydra thrust angle
Y_API uint32_t Y_CALL yVehicle_GetHydraThrustAngle(yVehicle const* vehicle);

/// Get train speed
Y_API float Y_CALL yVehicle_GetTrainSpeed(yVehicle const* vehicle);

/// Get last driver's pool id
Y_API int Y_CALL yVehicle_GetLastDriverPoolID(yVehicle const* vehicle);

/// Returns the current driver of the vehicle
Y_API yPlayer* Y_CALL yVehicle_GetDriver(yVehicle* vehicle);

/// Returns the number of passengers in the vehicle
Y_API size_t Y_CALL yVehicle_GetNumPassengers(yVehicle* vehicle);

/// Gets the passengers in the vehicle
Y_API size_t Y_CALL yVehicle_GetPassengers(yVehicle* vehicle, yPlayer** players);

/// Get the player's vehicle
/// Returns nullptr if they aren't in a vehicle
Y_API yVehicle* Y_CALL yPlayer_GetVehicle(yPlayer* player);

/// Reset player's vehicle value interally
/// *** DO NOT USE THIS IF YOU HAVE NO IDEA WHAT IT DOES
/// IT IS NOT FOR VEHICLE DESTRUCTION OR REMOVE PLAYER FROM ONE ***
Y_API void Y_CALL yPlayer_ResetVehicle(yPlayer* player);

/// Get the player's seat
/// Returns -1 if they aren't in a vehicle.
Y_API int Y_CALL yPlayer_GetSeat(yPlayer* player);

/// Checks if player is in a mod shop
Y_API bool Y_CALL yPlayer_IsInModShop(yPlayer* player);

/// Check if passenger is in drive-by mode or not
Y_API bool Y_CALL yPlayer_IsInDriveByMode(yPlayer* player);

/// Check if passenger is cuffed or not
Y_API bool Y_CALL yPlayer_IsCuffed(yPlayer* player);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
