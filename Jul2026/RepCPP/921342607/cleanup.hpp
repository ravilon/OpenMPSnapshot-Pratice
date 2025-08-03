/* ============================================================================ *
* SPC Integration - Plugin for SA-MP (San Andreas Multiplayer)                 *
* ============================================================================ *
*                                                                              *
* Copyright (c) 2025, SPC (SA-MP Programming Community)                        *
* All rights reserved.                                                         *
*                                                                              *
* Developed by: Calasans                                                       *
* Provided by: SA-MP Programming Community                                     *
* Repository: https://github.com/spc-samp/spc-integration                      *
*                                                                              *
* ============================================================================ *
*                                                                              *
* Licensed under the Apache License, Version 2.0 (the "License");              *
* you may not use this file except in compliance with the License.             *
* You may obtain a copy of the License at:                                     *
*                                                                              *
*     http://www.apache.org/licenses/LICENSE-2.0                               *
*                                                                              *
* Unless required by applicable law or agreed to in writing, software          *
* distributed under the License is distributed on an "AS IS" BASIS,            *
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.     *
* See the License for the specific language governing permissions and          *
* limitations under the License.                                               *
*                                                                              *
* ============================================================================ */

#pragma once

#include <atomic>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <signal.h>
#endif
//
#include "logger.hpp"
#include "server_info.hpp"
#include "amx_defines.h"

// AMX externs will not be included here, as "amx_defines.h" already includes them.
//
extern Server_Info SPC_Server_Info;

class Cleanup {
public:
static Cleanup& Get_Instance() {
static Cleanup instance;
return instance;
}

void Initialize() {
if (initialized.exchange(true))
return (void)Logger::Log_Formatted(Log_Level::WARNING, "Cleanup already initialized.", true);

Logger::Log_Formatted(Log_Level::INFO, "Initializing Cleanup signal handlers...");
Register_Signal_Handlers();
}

void Perform_Cleanup();

bool Is_Shutting_Down() const {
return is_shutting_down.load();
}
private:
Cleanup() = default;
~Cleanup() = default;
Cleanup(const Cleanup&) = delete;
Cleanup& operator=(const Cleanup&) = delete;

std::atomic<bool> initialized{ false };
std::atomic<bool> cleaned_up{ false };
std::atomic<bool> is_shutting_down{ false };

void Perform_Delete_Action();
void Reset_Global_Variables();
void Disable_Logger();
void Register_Signal_Handlers();

#if defined(_WIN32)
void Cleanup_Network_Resources();
static BOOL WINAPI Signal_Handler_Windows(DWORD);
#elif defined(__linux__)
void Cleanup_SSL_Resources();
static void Signal_Handler_Linux(int);
#endif
};