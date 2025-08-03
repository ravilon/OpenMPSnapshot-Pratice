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
#include <string_view>
#include <filesystem>

namespace spc_sysf = std::filesystem;

namespace Hash {
    std::string Generate_New_Hash();
    bool Save_Hash(const std::string& hash_to_save);
    std::string Load_Hash();
    bool Delete_Hash_File();
    bool Verify_Hash_File(std::string_view expected_hash);
    spc_sysf::path Get_Hash_File_Full_Path();
    std::string Toggle_Hash(std::string_view data);
}