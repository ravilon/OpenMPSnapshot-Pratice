/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_LIBRARY_MODE_H
#define NANOS6_LIBRARY_MODE_H


#include <stddef.h>

#include "major.h"


#pragma GCC visibility push(default)


// NOTE: The full version depends also on nanos6_major_api
//       That is:   nanos6_major_api . nanos6_library_mode_api
enum nanos6_library_mode_api_t { nanos6_library_mode_api = 3 };


#ifdef __cplusplus
extern "C" {
#endif

//! \brief Initialize the runtime in library mode
//! 
//! \returns NULL if successful, otherwise a string that describes the error
//! 
//! NOTE: this function is not defined in the loader. Instead it is defined in nanos6-library-mode.o which
//! needs to be linked with the application to correctly check if the API versions match.
__attribute__ ((used)) char const * nanos6_library_mode_init(void);


//! \brief Spawn asynchronously a function
//! 
//! \param function the function to be spawned
//! \param args a parameter that is passed to the function
//! \param completion_callback an optional function that will be called when the function finishes
//! \param completion_args a parameter that is passed to the completion callback
//! \param label an optional name for the function
void nanos6_spawn_function(
	void (*function)(void *),
	void *args,
	void (*completion_callback)(void *),
	void *completion_args,
	char const *label
);


//! \brief Spawn a function to be executed within a stream
//!
//! \param[in] function the function to be spawned
//! \param[in] args a parameter that is passed to the function
//! \param[in] callback an (optional) function to call when the function finishes
//! \param[in] callback_args parameters passed to the completion callback
//! \param[in] label an optional name for the function
//! \param[in] stream_id the identifier of a stream in which the function will
//! be executed
void nanos6_stream_spawn_function(
	void (*function)(void *),
	void *args,
	void (*callback)(void *),
	void *callback_args,
	char const *label,
	size_t stream_id
);


#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop


#endif /* NANOS6_LIBRARY_MODE_H */
