/*-------------------------------------------------------------------
 *  Copyright (c) 2025 Maicol Castro <maicolcastro.abc@gmail.com>.
 *  All rights reserved.
 *
 *  Distributed under the BSD 3-Clause License.
 *  See LICENSE.txt in the root directory of this project or at
 *  https://opensource.org/license/bsd-3-clause.
 *-----------------------------------------------------------------*/

#pragma once

#include "Common.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef enum {
    Y_DIALOG_STYLE_MSGBOX,
    Y_DIALOG_STYLE_INPUT,
    Y_DIALOG_STYLE_LIST,
    Y_DIALOG_STYLE_PASSWORD,
    Y_DIALOG_STYLE_TABLIST,
    Y_DIALOG_STYLE_TABLIST_HEADERS
} yDialogStyle;

typedef struct {
    void (Y_CALL* OnDialogResponse)(yPlayer* player, int dialogId, int response, int listItem, yStringView inputText);
} yPlayerDialogEventHandler;

Y_API void Y_CALL yAddDialogEventHandler(yPlayerDialogEventHandler const* handler);

/// Hide a dialog from a player
Y_API void Y_CALL yPlayer_HideDialog(yPlayer* player);

/// Show a dialog to a player
Y_API void Y_CALL yPlayer_ShowDialog(yPlayer* player, int id, yDialogStyle style, yStringView title, yStringView body, yStringView button1, yStringView button2);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
