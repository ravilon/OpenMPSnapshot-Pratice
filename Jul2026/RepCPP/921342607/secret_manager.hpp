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

#include "libraries/Dralyxor/dralyxor.hpp"

namespace Secret_Manager {
    namespace {
        inline auto& Get_Obfuscated_API_KEY() {
            static auto& api_key_obfuscated = DRALYXOR_KEY("YOUR_SECURE_KEY_TO_ACCESS_PHP", "KEY_THAT_ONLY_YOU_KNOW");
            
            return api_key_obfuscated;
        }

        inline auto& Get_Obfuscated_Backend_Host() {
            static auto& backend_host_obfuscated = DRALYXOR_KEY("example.com", "KEY_THAT_ONLY_YOU_KNOW");
            
            return backend_host_obfuscated;
        }

        inline auto& Get_Obfuscated_Backend_Path() {
            static auto& backend_path_obfuscated = DRALYXOR_KEY("/path/spc-integration.php", "KEY_THAT_ONLY_YOU_KNOW");
            
            return backend_path_obfuscated;
        }

        inline auto& Get_Obfuscated_Backend_API_Path() {
            static auto& backend_api_path_obfuscated = DRALYXOR_KEY("/path/api.php", "KEY_THAT_ONLY_YOU_KNOW");
            
            return backend_api_path_obfuscated;
        }

        inline auto& Get_Obfuscated_Hash_KEY() {
            static auto& key_obfuscated = DRALYXOR_KEY("YOUR_SECURE_KEY_TO_HASH", "KEY_THAT_ONLY_YOU_KNOW");
            
            return key_obfuscated;
        }
    }

    namespace Detail {
        inline auto& Get_API_KEY() {
            return Get_Obfuscated_API_KEY();
        }

        inline auto& Get_Backend_Host() {
            return Get_Obfuscated_Backend_Host();
        }

        inline auto& Get_Backend_Path() {
            return Get_Obfuscated_Backend_Path();
        }

        inline auto& Get_Backend_API_Path() {
            return Get_Obfuscated_Backend_API_Path();
        }

        inline auto& Get_Hash_KEY() {
            return Get_Obfuscated_Hash_KEY();
        }
    }
}