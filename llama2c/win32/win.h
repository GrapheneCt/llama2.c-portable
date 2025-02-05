#ifndef _WIN_H_
#define _WIN_H_

#define WIN32_LEAN_AND_MEAN      // Exclude rarely-used stuff from Windows headers
#include <windows.h>
#include <time.h>
#include <stdint.h>

#define ssize_t int64_t
#define ftell _ftelli64

#ifndef _OPENMP
#error Please compile with OpenMP support enabled on this platform
#endif

#ifdef __cplusplus
extern "C" {
#endif

//unsigned long long time_in_ms();
#define time_in_ms() GetTickCount() // return time in milliseconds, for benchmarking the model speed

// allocate/free main transformer memory
int allocate_transformer_memory(void** ptr, ssize_t size);
void free_transformer_memory(int id, void* ptr);

#ifdef __cplusplus
};
#endif

#endif /*  _WIN_H_ */
