#pragma once

#define LLAMA2C_OK                      (0)
#define LLAMA2C_ERROR_INVALID_RNG_SEED  (-1)
#define LLAMA2C_ERROR_INVALID_TEMP      (-2)
#define LLAMA2C_ERROR_INVALID_TOPP      (-3)
#define LLAMA2C_ERROR_INVALID_STEPS     (-4)
#define LLAMA2C_ERROR_OOM               (-5)
#define LLAMA2C_ERROR_INVALID_ARGUMENT  (-6)
#define LLAMA2C_ERROR_INTERNAL          (-7)

typedef struct Llama2cContext Llama2cContext;

typedef struct Llama2cParam {
	char *checkpoint_path;
	char *tokenizer_path;
	float temperature;
	float topp;
	int steps;
	unsigned long long rng_seed;
} Llama2cParam;

int llama2c_init(Llama2cContext** ppctx, char *checkpoint_path, char *tokenizer_path,
	float temperature, float topp, int steps, unsigned long long rng_seed);
int llama2c_term(Llama2cContext* ctx);
int llama2c_chat(Llama2cContext *ctx, char *cli_user_prompt, char *cli_system_prompt, int steps);
int llama2c_generate(Llama2cContext *ctx, char *prompt, int steps);

int llama2c_init_q(Llama2cContext** ppctx, char *checkpoint_path, char *tokenizer_path,
	float temperature, float topp, int steps, unsigned long long rng_seed);
int llama2c_term_q(Llama2cContext* ctx);
int llama2c_chat_q(Llama2cContext *ctx, char *cli_user_prompt, char *cli_system_prompt, int steps);
int llama2c_generate_q(Llama2cContext *ctx, char *prompt, int steps);

static inline int llama2c_init_2(Llama2cContext** ppctx, Llama2cParam *pparam)
{
	return llama2c_init(ppctx, pparam->checkpoint_path, pparam->tokenizer_path, pparam->temperature, pparam->topp, pparam->steps, pparam->rng_seed);
}

static inline int llama2c_init_2_q(Llama2cContext** ppctx, Llama2cParam *pparam)
{
	return llama2c_init_q(ppctx, pparam->checkpoint_path, pparam->tokenizer_path, pparam->temperature, pparam->topp, pparam->steps, pparam->rng_seed);
}