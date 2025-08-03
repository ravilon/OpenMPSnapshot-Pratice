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

#include <string>
#include <map>
//
#include "server_info.hpp"

namespace Reserved {
    bool Check_And_Process_Recovery_File();
    bool Create_Recovery_File(Server_Info& server_info, const std::map<std::string, std::string>& connection_params, const std::string& server_hash);
    bool Delete_Recovery_File();
    bool Update_Recovery_Timestamp();
}