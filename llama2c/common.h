#ifndef _COMMON_H_
#define _COMMON_H_

#include "platform.h"
#include <llama2c.h>

typedef struct {
	int dim; // transformer dimension
	int hidden_dim; // for ffn layers
	int n_layers; // number of layers
	int n_heads; // number of query heads
	int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
	int vocab_size; // vocabulary size, usually 256 (byte-level)
	int seq_len; // max sequence length
} Config;

typedef struct {
	char *str;
	int id;
} TokenIndex;

typedef struct {
	char** vocab;
	float* vocab_scores;
	TokenIndex *sorted_vocab;
	int vocab_size;
	unsigned int max_token_length;
	unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

typedef struct {
	float prob;
	int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
	int vocab_size;
	ProbIndex* probindex; // buffer used in top-p sampling
	float temperature;
	float topp;
	unsigned long long rng_state;
} Sampler;



void rmsnorm(float* o, float* x, float* weight, int size);
void softmax(float* x, int size);
int compare_tokens(const void *a, const void *b);
void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size);
void free_tokenizer(Tokenizer* t);
char* decode(Tokenizer* t, int prev_token, int token);
void safe_printf(char *piece);
int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size);
void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens);
int sample_argmax(float* probabilities, int n);
int sample_mult(float* probabilities, int n, float coin);
int compare(const void* a, const void* b);
int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin);
void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed);
void free_sampler(Sampler* sampler);
unsigned int random_u32(unsigned long long *state);
float random_f32(unsigned long long *state);
int sample(Sampler* sampler, float* logits);

void read_stdin(const char* guide, char* buffer, size_t bufsize);

#endif
