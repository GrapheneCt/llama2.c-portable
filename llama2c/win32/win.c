#include "win32/win.h"
#include "platform.h"
#include <stdlib.h>

int allocate_transformer_memory(void** ptr, ssize_t size)
{
    *ptr = malloc(size);
    return 1;
}

void free_transformer_memory(int id, void* ptr)
{
    free(ptr);
}

int llama2c_platform_thread_create_and_start(llama2c_platform_thread_entry entry, int cpu_affinity, unsigned int arg_size, const void *p_arg_block)
{
    return -1;
}

void llama2c_platform_thread_delete_self()
{
    
}

void llama2c_platform_thread_join(int id)
{

}

int llama2c_platform_event_flag_create()
{
    return -1;
}

void llama2c_platform_event_flag_delete(int id)
{

}

void llama2c_platform_event_flag_set(int id, unsigned int pattern)
{

}

void llama2c_platform_event_flag_wait_and_clear(int id, unsigned int pattern, unsigned int *p_timeout)
{

}