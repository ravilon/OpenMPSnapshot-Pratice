#pragma once


// Enum of all possible events.
typedef enum {
    TUI_NO_EVENT,
    TUI_QUIT_EVENT,
    TUI_RESTART_EVENT
} TUIEvent;

typedef struct {
    const char* ex_policy;
} TUITitleInfo;


// Width and height of each frame.
int tui_init(unsigned int* width, unsigned int* height, TUITitleInfo* title_info);
// Deinit trenderer.
void tui_deinit(void);
// Renders a frame onto the terminal.
// The frame must have the dimensions specified in trenderer_init and the valuse must be between 0.0 and 1.0.
void tui_render(float* frame);
// Get the last event.
TUIEvent tui_get_event(void);
