#ifndef _PSP2_H_
#define _PSP2_H_

#include <kernel.h>
#include <ult.h>
#include <stdint.h>

#define ssize_t int64_t

#ifdef __cplusplus
extern "C" {
#endif

unsigned long long time_in_ms();

// allocate/free main transformer memory
int allocate_transformer_memory(void** ptr, ssize_t size);
void free_transformer_memory(int id, void* ptr);

#ifdef __cplusplus
};
#endif

#endif /*  _PSP2_H_ */
