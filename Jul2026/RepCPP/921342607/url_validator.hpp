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
#include <chrono>

class URL_Validator {
    public:
        static bool Is_Valid(const std::string& url, const std::string& param_name = "", const std::string& validation_type = "");
    private:
        static bool Validate_With_Timeout(const std::string& url, const std::string& param_name, const std::string& validation_type);
        static bool Perform_Full_Validation(const std::string& url, const std::string& param_name, const std::string& validation_type);

        static bool Has_Valid_Scheme(const std::string& url);
        static bool Has_Valid_Host_And_Path(const std::string& url);
        static bool Has_Valid_Characters(const std::string& url);

        static bool Validate_Content_Accessibility_And_Type(const std::string& url, const std::string& param_name, const std::string& validation_type);
        static bool Check_Image_Magic_Numbers(const std::vector<unsigned char>& image_header_data, const std::string& claimed_content_type, const std::string& param_name);
};