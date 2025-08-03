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
#include <chrono>
//
#include "libraries/nlohmann/json.hpp"
//
#include "http_client.hpp"
#include "logger.hpp"
#include "plugin_constants.hpp"

namespace Plugin_Update {
inline bool Check_For_Update() {
const std::string host = "api.github.com";
const std::string path = "/repos/spc-samp/spc-integration/releases/latest";

Generic_HTTP_Response response = HTTP_Client::Perform_Generic_Request(host, path, true, "GET", {}, {}, std::chrono::milliseconds(5000));

if (!response.request_ok || response.status_code != 200)
return false;

try {
std::string response_body(response.body.begin(), response.body.end());
nlohmann::json release_info = nlohmann::json::parse(response_body);

if (!release_info.is_object() || !release_info.contains("tag_name"))
return false;

std::string latest_tag = release_info["tag_name"].get<std::string>();

if (!latest_tag.empty() && (latest_tag[0] == 'v' || latest_tag[0] == 'V'))
latest_tag.erase(0, 1);

int current_major = 0, current_minor = 0, current_patch = 0;
int latest_major = 0, latest_minor = 0, latest_patch = 0;
int parse_current_ok, parse_latest_ok;

#if defined(_MSC_VER)
parse_current_ok = sscanf_s(PLUGIN_VERSION, "%d.%d.%d", &current_major, &current_minor, &current_patch) == 3;
parse_latest_ok = sscanf_s(latest_tag.c_str(), "%d.%d.%d", &latest_major, &latest_minor, &latest_patch) == 3;
#else
parse_current_ok = sscanf(PLUGIN_VERSION, "%d.%d.%d", &current_major, &current_minor, &current_patch) == 3;
parse_latest_ok = sscanf(latest_tag.c_str(), "%d.%d.%d", &latest_major, &latest_minor, &latest_patch) == 3;
#endif

if (!parse_current_ok || !parse_latest_ok)
return false;

if (latest_major > current_major || (latest_major == current_major && latest_minor > current_minor) || (latest_major == current_major && latest_minor == current_minor && latest_patch > current_patch)) {
Logger::Log_Section_Divider("UPDATE AVAILABLE");

Logger::Log(Log_Level::WARNING, "A new version of the SPC Integration plugin ('v%s') is available (you are using 'v%s').", latest_tag.c_str(), PLUGIN_VERSION);
Logger::Log_Formatted(Log_Level::WARNING, "Please download the latest version from the official repository to ensure compatibility and access to possible new features.");
Logger::Log_Formatted(Log_Level::WARNING, "Repository: https://github.com/spc-samp/spc-integration/releases/latest");
Logger::Log_Formatted(Log_Level::ERROR_s, "Plugin loading has been halted. Please update the plugin.", true);

return true;
}

return false;
}
catch (const nlohmann::json::parse_error&) {
return false;
}
}
}