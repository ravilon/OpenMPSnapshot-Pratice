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
#include <filesystem>
#include <map>
#include <chrono>
//
#include "libraries/nlohmann/json.hpp"
//
#include "plugin_constants.hpp"
#include "logger.hpp"

namespace spc_sysf = std::filesystem;

namespace Manage_JSON {
using json = nlohmann::ordered_json;

enum class Load_Status {
SUCCESS,
FILE_NOT_FOUND,
PARSE_ERROR,
RECREATED_DEFAULT,
FAILED_TO_OPEN,
FAILED_TO_RECREATE
};

struct Load_Result {
json data;
Load_Status status;
bool was_corrupted = false;
};

Load_Result Load_JSON_File(const spc_sysf::path& file_path, bool allow_recreate = false, bool is_config_file = false);
bool Save_JSON_File(const spc_sysf::path& file_path, const json& data_to_save);

bool Update_JSON_Timestamp(const spc_sysf::path& file_path, const std::string& key, const std::string& timestamp_str);

json Create_Default_Config_JSON();
void Populate_Connection_Section(json& connection_json, const std::map<std::string, std::string>* initial_params = nullptr);
void Populate_Public_Info_Section(json& public_info_json, const std::map<std::string, std::string>* initial_params = nullptr);

template <typename T>

T Get_Value_From_JSON(const json& j, const std::string& key, const T& default_value) {
if (j.contains(key)) {
try {
return j.at(key).get<T>();
}
catch (const nlohmann::json::exception& e) {
return (Logger::Log(Log_Level::WARNING, "JSON parsing error for key '%s': %s. Returning default value.", key.c_str(), e.what()), default_value);
}
}

return default_value;
}

bool Get_Bool_From_JSON(const json& j, const std::string& key, bool default_value);
int Get_Int_From_JSON(const json& j, const std::string& key, int default_value);
json Get_Array_From_JSON(const json& j, const std::string& key);
}