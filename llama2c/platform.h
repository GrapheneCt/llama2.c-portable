#ifndef _PLATFORM_H_
#define _PLATFORM_H_

#ifdef WIN32
#include "win32/win.h"
#elif __psp2__
#include "psp2/psp2.h"
#else
#error Current platform is not supported!
#endif

#define LLAMA2C_PLATFORM_CPU_AFFINITY_AUTO (4096)

typedef int(*llama2c_platform_thread_entry)(unsigned int arg_size, void *p_arg_block);

int llama2c_platform_thread_create_and_start(llama2c_platform_thread_entry entry, int cpu_affinity, unsigned int arg_size, const void *p_arg_block);
void llama2c_platform_thread_delete_self();
void llama2c_platform_thread_join(int id);

int llama2c_platform_event_flag_create();
void llama2c_platform_event_flag_delete(int id);
void llama2c_platform_event_flag_set(int id, unsigned int pattern);
void llama2c_platform_event_flag_wait_and_clear(int id, unsigned int pattern, unsigned int *p_timeout);

#endif
