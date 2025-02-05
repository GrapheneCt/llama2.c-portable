/* Inference for Llama-2 Transformer model in pure C, int8 quantized forward pass. */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "common.h"
// ----------------------------------------------------------------------------
// Globals
int GS = 0; // group size global for quantization of the weights

// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int8_t* q;    // quantized values
    float* s; // scaling factors
} QuantizedTensor;

typedef struct {
    // token embedding table
    QuantizedTensor *q_tokens; // (vocab_size, dim)
    float* token_embedding_table; // same, but dequantized

    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    QuantizedTensor *wq; // (layer, dim, n_heads * head_size)
    QuantizedTensor *wk; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wv; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    QuantizedTensor *w1; // (layer, hidden_dim, dim)
    QuantizedTensor *w2; // (layer, dim, hidden_dim)
    QuantizedTensor *w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    QuantizedTensor *wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    QuantizedTensor xq; // quantized x (dim,)
    QuantizedTensor hq; // quantized hb (hidden_dim,)
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
    s->xq = (QuantizedTensor) { .q = calloc(p->dim, sizeof(int8_t)), .s = calloc(p->dim, sizeof(float)) };
    s->hq = (QuantizedTensor) { .q = calloc(p->hidden_dim, sizeof(int8_t)), .s = calloc(p->hidden_dim, sizeof(float)) };
    s->q = calloc(p->dim, sizeof(float));
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->k || !s->v || !s->att || !s->logits || !s->key_cache
     || !s->value_cache) {
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
    free(s->xq.q);
    free(s->xq.s);
    free(s->hq.q);
    free(s->hq.s);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

// ----------------------------------------------------------------------------
// Quantization functions

static void dequantize(QuantizedTensor *qx, float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = qx->q[i] * qx->s[i / GS];
    }
}

static void quantize(QuantizedTensor *qx, float* x, int n) {
    int num_groups = n / GS;
    float Q_MAX = 127.0f;

    for (int group = 0; group < num_groups; group++) {

        // find the max absolute value in the current group
        float wmax = 0.0;
        for (int i = 0; i < GS; i++) {
            float val = fabs(x[group * GS + i]);
            if (val > wmax) {
                wmax = val;
            }
        }

        // calculate and write the scaling factor
        float scale = wmax / Q_MAX;
        qx->s[group] = scale;

        // calculate and write the quantized values
        for (int i = 0; i < GS; i++) {
            float quant_value = x[group * GS + i] / scale; // scale
            int8_t quantized = (int8_t) round(quant_value); // round and clamp
            qx->q[group * GS + i] = quantized;
        }
    }
}

/* initialize `n` x quantized tensor (with `size_each` elements), starting from memory pointed at *ptr */
static QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each) {
    void *p = *ptr;
    QuantizedTensor *res = malloc(n * sizeof(QuantizedTensor));
    for(int i=0; i<n; i++) {
        /* map quantized int8 values*/
        res[i].q = (int8_t*)p;
        p = (int8_t*)p + size_each;
        /* map scale factors */
        res[i].s = (float*)p;
        p = (float*)p + size_each / GS;
    }
    *ptr = p; // advance ptr to current position
    return res;
}

static void memory_map_weights(TransformerWeights *w, Config* p, void* ptr, uint8_t shared_classifier) {
    int head_size = p->dim / p->n_heads;
    // first are the parameters that are kept in fp32 (the rmsnorm (1D) weights)
    float* fptr = (float*) ptr; // cast our pointer to float*
    w->rms_att_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_ffn_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_final_weight = fptr;
    fptr += p->dim;

    // now read all the quantized weights
    ptr = (void*)fptr; // now cast the pointer back to void*
    w->q_tokens = init_quantized_tensors(&ptr, 1, p->vocab_size * p->dim);
    // dequantize token embedding table
    w->token_embedding_table = malloc(p->vocab_size * p->dim * sizeof(float));
    dequantize(w->q_tokens, w->token_embedding_table, p->vocab_size * p->dim);

    w->wq = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_heads * head_size));
    w->wk = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wv = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wo = init_quantized_tensors(&ptr, p->n_layers, (p->n_heads * head_size) * p->dim);

    w->w1 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);
    w->w2 = init_quantized_tensors(&ptr, p->n_layers, p->hidden_dim * p->dim);
    w->w3 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);

    w->wcls = shared_classifier ? w->q_tokens : init_quantized_tensors(&ptr, 1, p->dim * p->vocab_size);
}

static void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* memblk, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in magic number (uint32), has to be 0x616b3432, i.e. "ak42" in ASCII
    uint32_t magic_number;
    if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (magic_number != 0x616b3432) { fprintf(stderr, "Bad magic number\n"); exit(EXIT_FAILURE); }
    // read in the version number (uint32), has to be 2
    int version;
    if (fread(&version, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (version != 2) { fprintf(stderr, "Bad version %d, need version 2\n", version); exit(EXIT_FAILURE); }
    int header_size = 256; // the header size for version 2 in bytes
    // read in the Config
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // read in flags
    uint8_t shared_classifier; // a byte to indicate if the classifier is shared
    if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1) { exit(EXIT_FAILURE); }
    int group_size; // the group size used in quantization
    if (fread(&group_size, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    GS = group_size; // set as global, as it will be used in many places
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    // memory map the Transformer weights into the data pointer
    fseek(file, 0, SEEK_SET);
    *memblk = allocate_transformer_memory(data, *file_size);
    if (*memblk < 0) { fprintf(stderr, "allocate_transformer_memory failed!\n"); exit(EXIT_FAILURE); }
    fread(*data, *file_size, 1, file);
    fclose(file);
    void* weights_ptr = ((char*)*data) + header_size; // skip header bytes. char is 1 byte
    memory_map_weights(weights, config, weights_ptr, shared_classifier);
}

static void build_transformer(Transformer *t, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->memblk, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

static void free_transformer(Transformer* t) {
    // free QuantizedTensors
    free(t->weights.q_tokens);
    free(t->weights.token_embedding_table);
    free(t->weights.wq);
    free(t->weights.wk);
    free(t->weights.wv);
    free(t->weights.wo);
    free(t->weights.w1);
    free(t->weights.w2);
    free(t->weights.w3);
    if(t->weights.wcls != t->weights.q_tokens) { free(t->weights.wcls); }
    // close the memory mapping
    if (t->memblk >= 0) { free_transformer_memory(t->memblk, t->data); }
    // free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

#ifndef _OPENMP
#define MAX_LIB_INSTANCES 1
#define THREAD_NUM 3

#define THREAD_LAUNCH_OFFSET (17)

typedef struct ThreadGlobals
{
    int evf;
    int threads[THREAD_NUM];

    int job;
    bool term_requested;

    float* xout;
    QuantizedTensor* x;
    QuantizedTensor* w;
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

ThreadGlobals thread_globals_q[MAX_LIB_INSTANCES];
int instance_num_q = 0;

int job_thrd_q(unsigned int arg_size, void *p_arg_block)
{
    unsigned short *arg = (unsigned short *)p_arg_block;
    int instance = arg[0];
    int idx = arg[1];

    while (1) {
        llama2c_platform_event_flag_wait_and_clear(thread_globals_q[instance].evf, 1 << (idx + THREAD_LAUNCH_OFFSET), NULL);

        if (thread_globals_q[instance].term_requested) {
            break;
        }

        if (thread_globals_q[instance].job == 0)
        {
            for (int i = thread_globals_q[instance].len * idx; i < thread_globals_q[instance].len * (idx + 1); i++) {
                float val = 0.0f;
                int32_t ival = 0;
                int in = i * thread_globals_q[instance].n;

                // do the matmul in groups of GS
                int j;
                for (j = 0; j <= thread_globals_q[instance].n - GS; j += GS) {
                    for (int k = 0; k < GS; k++) {
                        ival += ((int32_t)thread_globals_q[instance].x->q[j + k]) * ((int32_t)thread_globals_q[instance].w->q[in + j + k]);
                    }
                    val += ((float)ival) * thread_globals_q[instance].w->s[(in + j) / GS] * thread_globals_q[instance].x->s[j / GS];
                    ival = 0;
                }

                thread_globals_q[instance].xout[i] = val;
            }
        }
        else
        {
            float precalc_sqrt = sqrtf(thread_globals_q[instance].head_size);

            for (int h = thread_globals_q[instance].len * idx; h < thread_globals_q[instance].len * (idx + 1); h++) {
                // get the query vector for this head
                float* q = thread_globals_q[instance].s->q + h * thread_globals_q[instance].head_size;
                // attention scores for this head
                float* att = thread_globals_q[instance].s->att + h * thread_globals_q[instance].seq_len;
                // iterate over all timesteps, including the current one
                for (int t = 0; t <= thread_globals_q[instance].pos; t++) {
                    // get the key vector for this head and at this timestep
                    float* k = thread_globals_q[instance].s->key_cache + thread_globals_q[instance].loff + t * thread_globals_q[instance].kv_dim + (h / thread_globals_q[instance].kv_mul) * thread_globals_q[instance].head_size;
                    // calculate the attention score as the dot product of q and k
                    float score = 0.0f;
                    for (int i = 0; i < thread_globals_q[instance].head_size; i++) {
                        score += q[i] * k[i];
                    }
                    score /= precalc_sqrt;
                    // save the score to the attention buffer
                    att[t] = score;
                }

                // softmax the scores to get attention weights, from 0..pos inclusively
                softmax(att, thread_globals_q[instance].pos + 1);

                // weighted sum of the values, store back into xb
                float* xb = thread_globals_q[instance].s->xb + h * thread_globals_q[instance].head_size;
                memset(xb, 0, thread_globals_q[instance].head_size * sizeof(float));
                for (int t = 0; t <= thread_globals_q[instance].pos; t++) {
                    // get the value vector for this head and at this timestep
                    float* v = thread_globals_q[instance].s->value_cache + thread_globals_q[instance].loff + t * thread_globals_q[instance].kv_dim + (h / thread_globals_q[instance].kv_mul) * thread_globals_q[instance].head_size;
                    // get the attention weight for this timestep
                    float a = att[t];
                    // accumulate the weighted value into xb
                    for (int i = 0; i < thread_globals_q[instance].head_size; i++) {
                        xb[i] += a * v[i];
                    }
                }
            }
        }

        llama2c_platform_event_flag_set(thread_globals_q[instance].evf, 1 << idx);
    }

    llama2c_platform_thread_delete_self();

    return 0;
}
#endif

static void matmul(int instance, float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // inputs to this function are both quantized

    int tmp = 0;

#ifndef _OPENMP
    tmp = d;
    while (tmp % THREAD_NUM != 0)
    {
        tmp--;
    }

    thread_globals_q[instance].xout = xout;
    thread_globals_q[instance].x = x;
    thread_globals_q[instance].w = w;
    thread_globals_q[instance].n = n;
    thread_globals_q[instance].len = d / THREAD_NUM;
    thread_globals_q[instance].job = 0;

    llama2c_platform_event_flag_set(thread_globals_q[instance].evf, ((1 << (THREAD_NUM)) - 1) << THREAD_LAUNCH_OFFSET);
    llama2c_platform_event_flag_wait_and_clear(thread_globals_q[instance].evf, (1 << THREAD_NUM) - 1, NULL);
#endif

    int i;
    #pragma omp parallel for private(i)
    for (i = tmp; i < d; i++) {

        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;

        // do the matmul in groups of GS
        int j;
        for (j = 0; j <= n - GS; j += GS) {
            for (int k = 0; k < GS; k++) {
                ival += ((int32_t) x->q[j + k]) * ((int32_t) w->q[in + j + k]);
            }
            val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
            ival = 0;
        }

        xout[i] = val;
    }
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
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    memcpy(x, w->token_embedding_table + token*dim, dim * sizeof(float));

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // qkv matmuls for this position
        quantize(&s->xq, s->xb, dim);
        matmul(instance, s->q, &s->xq, w->wq + l, dim, dim);
        matmul(instance, s->k, &s->xq, w->wk + l, dim, kv_dim);
        matmul(instance, s->v, &s->xq, w->wv + l, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // save key,value at this time step (pos) to our kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        float* value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));

        // multihead attention. iterate over all heads

        int tmp = 0;

#ifndef _OPENMP
        tmp = p->n_heads;
        while (tmp % THREAD_NUM != 0)
        {
            tmp--;
        }

        thread_globals_q[instance].s = s;
        thread_globals_q[instance].head_size = head_size;
        thread_globals_q[instance].pos = pos;
        thread_globals_q[instance].loff = loff;
        thread_globals_q[instance].kv_dim = kv_dim;
        thread_globals_q[instance].kv_mul = kv_mul;
        thread_globals_q[instance].seq_len = p->seq_len;
        thread_globals_q[instance].len = tmp / THREAD_NUM;
        thread_globals_q[instance].job = 1;

        llama2c_platform_event_flag_set(thread_globals_q[instance].evf, ((1 << (THREAD_NUM)) - 1) << THREAD_LAUNCH_OFFSET);
        llama2c_platform_event_flag_wait_and_clear(thread_globals_q[instance].evf, (1 << THREAD_NUM) - 1, NULL);
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
        quantize(&s->xq, s->xb, dim);
        matmul(instance, s->xb2, &s->xq, w->wo + l, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        quantize(&s->xq, s->xb, dim);
        matmul(instance, s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        matmul(instance, s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);

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
        quantize(&s->hq, s->hb, hidden_dim);
        matmul(instance, s->xb, &s->hq, w->w2 + l, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    quantize(&s->xq, x, dim);
    matmul(instance, s->logits, &s->xq, w->wcls, dim, p->vocab_size);
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

int llama2c_generate_q(Llama2cContext *ctx, char *prompt, int steps) {

    if (!ctx) return LLAMA2C_ERROR_INVALID_ARGUMENT;

    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt) + 3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
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

        // advance the state state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        }
        else {
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

int llama2c_chat_q(Llama2cContext *ctx, char *cli_user_prompt, char *cli_system_prompt, int steps) {

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
                }
                else {
                    // system prompt was passed in, use it
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            // get the user prompt
            if (pos == 0 && cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt, cli_user_prompt);
            }
            else {
                // otherwise get user prompt from stdin
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            // render user/system prompts into the Llama 2 Chat schema
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            }
            else {
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
        }
        else {
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

int llama2c_init_q(Llama2cContext** ppctx, char *checkpoint_path, char *tokenizer_path,
    float temperature, float topp, int steps, unsigned long long rng_seed) {

    // parameter validation/overrides
    if (rng_seed == 0) return LLAMA2C_ERROR_INVALID_RNG_SEED;
    if (temperature < 0.0) return LLAMA2C_ERROR_INVALID_TEMP;
    if (topp < 0.0 || 1.0 < topp) return LLAMA2C_ERROR_INVALID_TOPP;
    if (steps <= 0) return LLAMA2C_ERROR_INVALID_STEPS;

#ifndef _OPENMP
    if (instance_num_q == MAX_LIB_INSTANCES) return LLAMA2C_ERROR_OOM;
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
    memset(&thread_globals_q[instance_num_q], 0, sizeof(ThreadGlobals));

    ctx->instance = instance_num_q;

    thread_globals_q[instance_num_q].evf = llama2c_platform_event_flag_create();

    unsigned short arg[2];
    arg[0] = instance_num_q;
    for (int i = 0; i < THREAD_NUM; i++) {
        arg[1] = i;
        thread_globals_q[instance_num_q].threads[i] = llama2c_platform_thread_create_and_start(job_thrd_q, i, sizeof(arg), (const void *)arg);
        if (thread_globals_q[instance_num_q].threads[i] < 0) {
            llama2c_term(ctx);
            return LLAMA2C_ERROR_INTERNAL;
        }
    }

    instance_num_q++;
#endif

    return LLAMA2C_OK;
}

int llama2c_term_q(Llama2cContext* ctx)
{
    if (!ctx) return LLAMA2C_ERROR_INVALID_ARGUMENT;

    // memory and file handles cleanup
    free_sampler(&ctx->sampler);
    free_tokenizer(&ctx->tokenizer);
    free_transformer(&ctx->transformer);

#ifndef _OPENMP
    thread_globals_q[ctx->instance].term_requested = true;

    llama2c_platform_event_flag_set(thread_globals_q[ctx->instance].evf, ((1 << (THREAD_NUM)) - 1) << THREAD_LAUNCH_OFFSET);

    for (int i = 0; i < THREAD_NUM; i++) {
        llama2c_platform_thread_join(thread_globals_q[ctx->instance].threads[i]);
    }

    llama2c_platform_event_flag_delete(thread_globals_q[ctx->instance].evf);

    instance_num_q--;
#endif

    free(ctx);

    return LLAMA2C_OK;
}
