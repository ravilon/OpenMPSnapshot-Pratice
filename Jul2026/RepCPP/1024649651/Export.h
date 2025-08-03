/*-------------------------------------------------------------------
*  Copyright (c) 2025 Maicol Castro <maicolcastro.abc@gmail.com>.
*  All rights reserved.
*
*  Distributed under the BSD 3-Clause License.
*  See LICENSE.txt in the root directory of this project or at
*  https://opensource.org/license/bsd-3-clause.
*-----------------------------------------------------------------*/

#pragma once

#if defined __cplusplus
#define Y_EXTERN_C extern "C"
#else
#define Y_EXTERN_C
#endif

#if defined __GNUC__
#define Y_EXPORT Y_EXTERN_C __attribute__((visibility("default")))
#define Y_IMPORT Y_EXTERN_C
#elif defined _MSC_VER
#define Y_EXPORT Y_EXTERN_C __declspec(dllexport)
#define Y_IMPORT Y_EXTERN_C __declspec(dllimport)
#else
#error unsupported compiler
#endif

#if defined _WIN32 || defined __CYGWIN__
#define Y_CALL __stdcall
#elif defined __unix__
#define Y_CALL
#else
#error unsupported platform
#endif
