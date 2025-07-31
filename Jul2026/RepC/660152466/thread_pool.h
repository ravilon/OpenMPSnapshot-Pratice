#pragma once

#include <stddef.h>



typedef struct ThreadPool ThreadPool;
typedef void (*TaskFn)(void *args);


ThreadPool* tp_init(size_t n_threads, size_t max_tasks);
void tp_deinit(ThreadPool* tp);
int tp_add_task(ThreadPool* tp, TaskFn f, void* args);
void tp_wait(ThreadPool* tp);