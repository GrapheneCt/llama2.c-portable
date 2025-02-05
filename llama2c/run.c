/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "common.h"
// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int memblk; // id of the memblock that stores the transformer
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

static void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

static void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

static void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

static void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* memblk, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    // memory map the Transformer weights into the data pointer
    fseek(file, 0, SEEK_SET);
    *memblk = allocate_transformer_memory(data, *file_size);
    if (*memblk < 0) { fprintf(stderr, "allocate_transformer_memory failed!\n"); exit(EXIT_FAILURE); }
    fread(*data, *file_size, 1, file);
    fclose(file);
    float* weights_ptr = *data + sizeof(Config)/sizeof(float);
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

static void build_transformer(Transformer *t, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->memblk, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

static void free_transformer(Transformer* t) {
    // close the memory mapping
    if (t->memblk >= 0) { free_transformer_memory(t->memblk, t->data); }
    // free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

#ifndef _OPENMP
#define MAX_LIB_INSTANCES 1
#define THREAD_NUM 4

#define THREAD_LAUNCH_OFFSET (17)

typedef struct ThreadGlobals
{
    int evf;
    int threads[THREAD_NUM];

    int job;
    bool term_requested;

    float* xout;
    float* x;
    float* w;
    int n;
    int len;

    RunState* s;
    int head_size;
    int pos;
    int loff;
    int kv_dim;
    int kv_mul;
    int seq_len;
} ThreadGlobals;

ThreadGlobals thread_globals[MAX_LIB_INSTANCES];
int instance_num = 0;

int job_thrd(unsigned int arg_size, void *p_arg_block)
{
    unsigned short *arg = (unsigned short *)p_arg_block;
    int instance = arg[0];
    int idx = arg[1];

    while (1) {
        llama2c_platform_event_flag_wait_and_clear(thread_globals[instance].evf, 1 << (idx + THREAD_LAUNCH_OFFSET), NULL);

        if (thread_globals[instance].term_requested) {
            break;
        }

        if (thread_globals[instance].job == 0)
        {
#ifdef __ARM_NEON__
            for (int i = thread_globals[instance].len * idx; i < thread_globals[instance].len * (idx + 1); i++) {
                float32x4_t val_vec = vdupq_n_f32(0.0f);  // Initialize NEON vector to 0
                int j = 0;

                // Vectorized loop with unrolling
                for (; j <= thread_globals[instance].n - 8; j += 8) {
                    // Prefetch next data for cache
                    __builtin_prefetch(&thread_globals[instance].w[i * thread_globals[instance].n + j + 8], 0, 1);
                    __builtin_prefetch(&thread_globals[instance].x[j + 8], 0, 1);

                    // Process first 4 elements
                    float32x4_t w_vec1 = vld1q_f32(&thread_globals[instance].w[i * thread_globals[instance].n + j]);   // Load 4 elements of w
                    float32x4_t x_vec1 = vld1q_f32(&thread_globals[instance].x[j]);           // Load 4 elements of x
                    val_vec = vmlaq_f32(val_vec, w_vec1, x_vec1);    // Fused multiply-add

                    // Process next 4 elements
                    float32x4_t w_vec2 = vld1q_f32(&thread_globals[instance].w[i * thread_globals[instance].n + j + 4]); // Load next 4 elements of w
                    float32x4_t x_vec2 = vld1q_f32(&thread_globals[instance].x[j + 4]);         // Load next 4 elements of x
                    val_vec = vmlaq_f32(val_vec, w_vec2, x_vec2);      // Fused multiply-add
                }

                // Handle remainder (if n isn't a multiple of 8)
                for (; j <= thread_globals[instance].n - 4; j += 4) {
                    float32x4_t w_vec = vld1q_f32(&thread_globals[instance].w[i * thread_globals[instance].n + j]);   // Load 4 elements of w
                    float32x4_t x_vec = vld1q_f32(&thread_globals[instance].x[j]);           // Load 4 elements of x
                    val_vec = vmlaq_f32(val_vec, w_vec, x_vec);     // Fused multiply-add
                }

                // Horizontal add for NEON vector
                float val = vgetq_lane_f32(val_vec, 0) +
                    vgetq_lane_f32(val_vec, 1) +
                    vgetq_lane_f32(val_vec, 2) +
                    vgetq_lane_f32(val_vec, 3);

                // Scalar loop for remaining elements
                for (; j < thread_globals[instance].n; j++) {
                    val += thread_globals[instance].w[i * thread_globals[instance].n + j] * thread_globals[instance].x[j];
                }

                thread_globals[instance].xout[i] = val;
            }
#else
            for (int i = thread_globals[instance].len * idx; i < thread_globals[instance].len * (idx + 1); i++) {
                float val = 0.0f;
                for (int j = 0; j < thread_globals[instance].n; j++) {
                    val += thread_globals[instance].w[i * thread_globals[instance].n + j] * thread_globals[instance].x[j];
                }
                thread_globals[instance].xout[i] = val;
            }
#endif
        }
        else
        {
            float precalc_sqrt = sqrtf(thread_globals[instance].head_size);

            for (int h = thread_globals[instance].len * idx; h < thread_globals[instance].len * (idx + 1); h++) {
                // get the query vector for this head
                float* q = thread_globals[instance].s->q + h * thread_globals[instance].head_size;
                // attention scores for this head
                float* att = thread_globals[instance].s->att + h * thread_globals[instance].seq_len;
                // iterate over all timesteps, including the current one
                for (int t = 0; t <= thread_globals[instance].pos; t++) {
                    // get the key vector for this head and at this timestep
                    float* k = thread_globals[instance].s->key_cache + thread_globals[instance].loff + t * thread_globals[instance].kv_dim + (h / thread_globals[instance].kv_mul) * thread_globals[instance].head_size;
                    // calculate the attention score as the dot product of q and k
                    float score = 0.0f;
                    for (int i = 0; i < thread_globals[instance].head_size; i++) {
                        score += q[i] * k[i];
                    }
                    score /= precalc_sqrt;
                    // save the score to the attention buffer
                    att[t] = score;
                }

                // softmax the scores to get attention weights, from 0..pos inclusively
                softmax(att, thread_globals[instance].pos + 1);

                // weighted sum of the values, store back into xb
                float* xb = thread_globals[instance].s->xb + h * thread_globals[instance].head_size;
                memset(xb, 0, thread_globals[instance].head_size * sizeof(float));
                for (int t = 0; t <= thread_globals[instance].pos; t++) {
                    // get the value vector for this head and at this timestep
                    float* v = thread_globals[instance].s->value_cache + thread_globals[instance].loff + t * thread_globals[instance].kv_dim + (h / thread_globals[instance].kv_mul) * thread_globals[instance].head_size;
                    // get the attention weight for this timestep
                    float a = att[t];
                    // accumulate the weighted value into xb
                    for (int i = 0; i < thread_globals[instance].head_size; i++) {
                        xb[i] += a * v[i];
                    }
                }
            }
        }

        llama2c_platform_event_flag_set(thread_globals[instance].evf, 1 << idx);
    }

    llama2c_platform_thread_delete_self();

    return 0;
}
#endif

static void matmul(int instance, float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function

    int tmp = 0;

#ifndef _OPENMP
    tmp = d;
    while (tmp % THREAD_NUM != 0)
    {
        tmp--;
    }

    thread_globals[instance].xout = xout;
    thread_globals[instance].x = x;
    thread_globals[instance].w = w;
    thread_globals[instance].n = n;
    thread_globals[instance].len = d / THREAD_NUM;
    thread_globals[instance].job = 0;

    llama2c_platform_event_flag_set(thread_globals[instance].evf, ((1 << (THREAD_NUM)) - 1) << THREAD_LAUNCH_OFFSET);
    llama2c_platform_event_flag_wait_and_clear(thread_globals[instance].evf, (1 << THREAD_NUM) - 1, NULL);
#endif

#ifdef __ARM_NEON__
    int i;
    #pragma omp parallel for private(i)
    for (i = tmp; i < d; i++) {
        float32x4_t val_vec = vdupq_n_f32(0.0f);  // Initialize NEON vector to 0
        int j = 0;

        // Vectorized loop with unrolling
        for (; j <= n - 8; j += 8) {
            // Prefetch next data for cache
            __builtin_prefetch(&w[i * n + j + 8], 0, 1);
            __builtin_prefetch(&x[j + 8], 0, 1);

            // Process first 4 elements
            float32x4_t w_vec1 = vld1q_f32(&w[i * n + j]);   // Load 4 elements of w
            float32x4_t x_vec1 = vld1q_f32(&x[j]);           // Load 4 elements of x
            val_vec = vmlaq_f32(val_vec, w_vec1, x_vec1);    // Fused multiply-add

            // Process next 4 elements
            float32x4_t w_vec2 = vld1q_f32(&w[i * n + j + 4]); // Load next 4 elements of w
            float32x4_t x_vec2 = vld1q_f32(&x[j + 4]);         // Load next 4 elements of x
            val_vec = vmlaq_f32(val_vec, w_vec2, x_vec2);      // Fused multiply-add
        }

        // Handle remainder (if n isn't a multiple of 8)
        for (; j <= n - 4; j += 4) {
            float32x4_t w_vec = vld1q_f32(&w[i * n + j]);   // Load 4 elements of w
            float32x4_t x_vec = vld1q_f32(&x[j]);           // Load 4 elements of x
            val_vec = vmlaq_f32(val_vec, w_vec, x_vec);     // Fused multiply-add
        }

        // Horizontal add for NEON vector
        float val = vgetq_lane_f32(val_vec, 0) +
            vgetq_lane_f32(val_vec, 1) +
            vgetq_lane_f32(val_vec, 2) +
            vgetq_lane_f32(val_vec, 3);

        // Scalar loop for remaining elements
        for (; j < n; j++) {
            val += w[i * n + j] * x[j];
        }

        xout[i] = val;
    }
#else
    int i;
    #pragma omp parallel for private(i)
    for (i = tmp; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
#endif
}

static float* forward(int instance, Transformer* transformer, int token, int pos) {

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(*x));

    // forward all the layers
    for (unsigned long long l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position
        matmul(instance, s->q, s->xb, w->wq + l * dim*dim, dim, dim);
        matmul(instance, s->k, s->xb, w->wk + l * dim*kv_dim, dim, kv_dim);
        matmul(instance, s->v, s->xb, w->wv + l * dim*kv_dim, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        // multihead attention. iterate over all heads

        int tmp = 0;

#ifndef _OPENMP
        tmp = p->n_heads;
        while (tmp % THREAD_NUM != 0)
        {
            tmp--;
        }

        thread_globals[instance].s = s;
        thread_globals[instance].head_size = head_size;
        thread_globals[instance].pos = pos;
        thread_globals[instance].loff = loff;
        thread_globals[instance].kv_dim = kv_dim;
        thread_globals[instance].kv_mul = kv_mul;
        thread_globals[instance].seq_len = p->seq_len;
        thread_globals[instance].len = tmp / THREAD_NUM;
        thread_globals[instance].job = 1;

        llama2c_platform_event_flag_set(thread_globals[instance].evf, ((1 << (THREAD_NUM)) - 1) << THREAD_LAUNCH_OFFSET);
        llama2c_platform_event_flag_wait_and_clear(thread_globals[instance].evf, (1 << THREAD_NUM) - 1, NULL);
#endif
        
        float precalc_sqrt = sqrtf(head_size);

        int h;
        #pragma omp parallel for private(h)
        for (h = tmp; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= precalc_sqrt;
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        matmul(instance, s->xb2, s->xb, w->wo + l * dim*dim, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(instance, s->hb, s->xb, w->w1 + l * dim*hidden_dim, dim, hidden_dim);
        matmul(instance, s->hb2, s->xb, w->w3 + l * dim*hidden_dim, dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        matmul(instance, s->xb, s->hb, w->w2 + l * dim*hidden_dim, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul(instance, s->logits, x, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// generation loop

struct Llama2cContext {
    Transformer transformer;
    Tokenizer tokenizer;
    Sampler sampler;
    int instance;
};

int llama2c_generate(Llama2cContext *ctx, char *prompt, int steps) {
    
    if (!ctx) return LLAMA2C_ERROR_INVALID_ARGUMENT;

    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(&ctx->tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    unsigned long long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        float* logits = forward(ctx->instance, &ctx->transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(&ctx->sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(&ctx->tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        unsigned long long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (double)(pos - 1) / ((double)(end - start) / 1000.0f));
    }

    free(prompt_tokens);

    return LLAMA2C_OK;
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

int llama2c_chat(Llama2cContext *ctx, char *cli_user_prompt, char *cli_system_prompt, int steps) {

    if (!ctx) return LLAMA2C_ERROR_INVALID_ARGUMENT;

    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are soomewhat haphazardly and unsafely set atm
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx;

    // start the main loop
    int8_t user_turn = 1; // user starts
    int next;        // will store the next token in the sequence
    int token;       // stores the current token to feed into the transformer
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            // get the (optional) system prompt at position 0
            if (pos == 0) {
                // at position 0, the user can also contribute a system prompt
                if (cli_system_prompt == NULL) {
                    // system prompt was not passed in, attempt to get it from stdin
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    // system prompt was passed in, use it
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            // get the user prompt
            if (pos == 0 && cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt, cli_user_prompt);
            } else {
                // otherwise get user prompt from stdin
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            // render user/system prompts into the Llama 2 Chat schema
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            // encode the rendered prompt into tokens
            encode(&ctx->tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = 0;
            printf("Assistant: ");
        }

        // determine the token to pass into the transformer next
        if (user_idx < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[user_idx++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }
        // EOS (=2) token ends the Assistant turn
        if (token == 2) { user_turn = 1; }

        // forward the transformer to get logits for the next token
        float* logits = forward(ctx->instance, &ctx->transformer, token, pos);
        next = sample(&ctx->sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
            // the Assistant is responding, so print its output
            char* piece = decode(&ctx->tokenizer, token, next);
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }
        if (next == 2) { printf("\n"); }
    }
    printf("\n");
    free(prompt_tokens);

    return LLAMA2C_OK;
}


// ----------------------------------------------------------------------------

int llama2c_init(Llama2cContext** ppctx, char *checkpoint_path, char *tokenizer_path,
    float temperature, float topp, int steps, unsigned long long rng_seed) {

    // parameter validation/overrides
    if (rng_seed == 0) return LLAMA2C_ERROR_INVALID_RNG_SEED;
    if (temperature < 0.0) return LLAMA2C_ERROR_INVALID_TEMP;
    if (topp < 0.0 || 1.0 < topp) return LLAMA2C_ERROR_INVALID_TOPP;
    if (steps <= 0) return LLAMA2C_ERROR_INVALID_STEPS;

#ifndef _OPENMP
    if (instance_num == MAX_LIB_INSTANCES) return LLAMA2C_ERROR_OOM;
#endif

    *ppctx = malloc(sizeof(Llama2cContext));
    Llama2cContext *ctx = *ppctx;
    if (!ctx) return LLAMA2C_ERROR_OOM;

    // build the Transformer via the model .bin file
    build_transformer(&ctx->transformer, checkpoint_path);
    if (steps == 0 || steps > ctx->transformer.config.seq_len) steps = ctx->transformer.config.seq_len; // override to ~max length

    // build the Tokenizer via the tokenizer .bin file
    build_tokenizer(&ctx->tokenizer, tokenizer_path, ctx->transformer.config.vocab_size);

    // build the Sampler
    build_sampler(&ctx->sampler, ctx->transformer.config.vocab_size, temperature, topp, rng_seed);

#ifndef _OPENMP
    memset(&thread_globals[instance_num], 0, sizeof(ThreadGlobals));

    ctx->instance = instance_num;

    thread_globals[instance_num].evf = llama2c_platform_event_flag_create();

    unsigned short arg[2];
    arg[0] = instance_num;
    for (int i = 0; i < THREAD_NUM; i++) {
        arg[1] = i;
        thread_globals[instance_num].threads[i] = llama2c_platform_thread_create_and_start(job_thrd, i, sizeof(arg), (const void *)arg);
        if (thread_globals[instance_num].threads[i] < 0) {
            llama2c_term(ctx);
            return LLAMA2C_ERROR_INTERNAL;
        }
    }

    instance_num++;
#endif

    return LLAMA2C_OK;
}

int llama2c_term(Llama2cContext* ctx)
{
    if (!ctx) return LLAMA2C_ERROR_INVALID_ARGUMENT;

    // memory and file handles cleanup
    free_sampler(&ctx->sampler);
    free_tokenizer(&ctx->tokenizer);
    free_transformer(&ctx->transformer);

#ifndef _OPENMP
    thread_globals[ctx->instance].term_requested = true;

    llama2c_platform_event_flag_set(thread_globals[ctx->instance].evf, ((1 << (THREAD_NUM)) - 1) << THREAD_LAUNCH_OFFSET);

    for (int i = 0; i < THREAD_NUM; i++) {
        llama2c_platform_thread_join(thread_globals[ctx->instance].threads[i]);
    }

    llama2c_platform_event_flag_delete(thread_globals[ctx->instance].evf);

    instance_num--;
#endif

    free(ctx);

    return LLAMA2C_OK;
}
