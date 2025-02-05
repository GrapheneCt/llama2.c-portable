#include "psp2/psp2.h"
#include "platform.h"
#include <stdlib.h>

unsigned int sceLibcHeapSize = 230 * 1024 * 1024;

unsigned long long time_in_ms()
{
	return sceKernelGetProcessTimeWide() / 1000;
}

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
	int ret = -1;
	unsigned int affinity = SCE_KERNEL_CPU_MASK_USER_ALL;

	if (cpu_affinity != LLAMA2C_PLATFORM_CPU_AFFINITY_AUTO)
	{
		affinity = (1 << cpu_affinity) << SCE_KERNEL_CPU_MASK_SHIFT;
	}

	ret = sceKernelCreateThread("llama2c_job", entry, SCE_KERNEL_INDIVIDUAL_QUEUE_HIGHEST_PRIORITY, SCE_KERNEL_4KiB, 0, affinity, NULL);
	if (ret < 0)
	{
		return ret;
	}

	if (sceKernelStartThread(ret, arg_size, p_arg_block) < 0)
	{
		return -1;
	}

	sceKernelDelayThread(5000);

	return ret;
}

void llama2c_platform_thread_delete_self()
{
	sceKernelExitDeleteThread(0);
}

void llama2c_platform_thread_join(int id)
{
	sceKernelWaitThreadEnd(id, NULL, NULL);
}

int llama2c_platform_event_flag_create()
{
	return sceKernelCreateEventFlag("llama2c_job_sync", SCE_KERNEL_EVF_ATTR_TH_FIFO | SCE_KERNEL_EVF_ATTR_MULTI, 0, NULL);
}

void llama2c_platform_event_flag_delete(int id)
{
	sceKernelDeleteEventFlag(id);
}

void llama2c_platform_event_flag_set(int id, unsigned int pattern)
{
	sceKernelSetEventFlag(id, pattern);
}

void llama2c_platform_event_flag_wait_and_clear(int id, unsigned int pattern, unsigned int *p_timeout)
{
	sceKernelWaitEventFlag(id, pattern, SCE_KERNEL_EVF_WAITMODE_AND | SCE_KERNEL_EVF_WAITMODE_CLEAR_PAT, NULL, p_timeout);
}