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

#if !defined(HAVE_STDINT_H)
#define HAVE_STDINT_H
#endif
//
#include "libraries/sdk/plugin.h"

typedef cell(AMX_NATIVE_CALL* AMX_Get_String_t)(char* Dest, const cell* Source, int Use_Wide_Char, size_t Size);
typedef int (AMX_NATIVE_CALL* AMX_Register_t)(AMX* Amx, const AMX_NATIVE_INFO* Native_List, int Number);
typedef int (AMX_NATIVE_CALL* AMX_Set_String_t)(cell* Dest, const char* Source, int Pack, int Use_Wide_Char, size_t Size);
typedef int (AMX_NATIVE_CALL* AMX_String_Len_t)(const cell* Cell_String, int* Length);
typedef int (AMX_NATIVE_CALL* AMX_Get_Addr_t)(AMX* Amx, cell Amx_Addr, cell** Phys_Addr);

extern AMX_Get_String_t AMX_Get_String_Ptr;
extern AMX_Register_t AMX_Register_Ptr;
extern AMX_Set_String_t AMX_Set_String_Ptr;
extern AMX_String_Len_t AMX_String_Len_Ptr;
extern AMX_Get_Addr_t AMX_Get_Addr_Ptr;