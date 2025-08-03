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
#include <vector>
#include <fstream>

enum class Log_Level {
INFO,
ERROR_s,
WARNING
};

class Logger {
public:
using Log_Printf_t = void (*)(const char* format, ...);
static void Initialize(Log_Printf_t log_func);

static void Log(Log_Level level, const char* format, ...);
static void Log_Formatted(Log_Level level, const std::string& message, bool skip_line = false);
static void Log_Multiline(Log_Level level, const std::vector<std::string>& messages, bool skip_line = false);
static void Log_Section_Divider(const std::string& section_name);
private:
static Log_Printf_t log_print_f;

static std::string Get_Level_Prefix(Log_Level level);
static void Write_To_Log_File(const std::string& message);
};