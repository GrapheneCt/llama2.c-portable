#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <llama2c.h>

//#define CLI
//#define PROMPT_FROM_FILE

void run(Llama2cParam *param, char *mode, char *prompt, char *system_prompt)
{
	Llama2cContext *ctx;
	int ret = LLAMA2C_OK;

	ret = llama2c_init_2(&ctx, param);
	if (ret != LLAMA2C_OK) {
		fprintf(stderr, "llama2c_init_2() failed: %d\n", ret);
		return;
	}

	// run!
	if (strcmp(mode, "generate") == 0) {
		ret = llama2c_generate(ctx, prompt, param->steps);
	}
	else if (strcmp(mode, "chat") == 0) {
		ret = llama2c_chat(ctx, prompt, system_prompt, param->steps);
	}

	if (ret != LLAMA2C_OK) {
		fprintf(stderr, "llama2c main API failed: %d\n", ret);
		return;
	}

	ret = llama2c_term(ctx);
	if (ret != LLAMA2C_OK) {
		fprintf(stderr, "llama2c_term() failed: %d\n", ret);
		return;
	}
}

#ifdef CLI

void error_usage() {
	fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
	fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
	fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
	fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
	fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
	fprintf(stderr, "  -i <string> input prompt\n");
	fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
	fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
	fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
	exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

	// default parameters
	char *checkpoint_path = NULL;  // e.g. out/model.bin
	char *tokenizer_path = "tokenizer.bin";
	float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
	float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
	int steps = 256;            // number of steps to run for
	char *prompt = NULL;        // prompt string
	unsigned long long rng_seed = time(NULL); // seed rng with time by default
	char *mode = "generate";    // generate|chat
	char *system_prompt = NULL; // the (optional) system prompt to use in chat mode

	// poor man's C argparse so we can override the defaults above from the command line
	if (argc >= 2) { checkpoint_path = argv[1]; }
	else { error_usage(); }
	for (int i = 2; i < argc; i += 2) {
		// do some basic validation
		if (i + 1 >= argc) { error_usage(); } // must have arg after flag
		if (argv[i][0] != '-') { error_usage(); } // must start with dash
		if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
		// read in the args
		if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
		else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
		else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
		else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
		else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
		else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
		else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
		else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
		else { error_usage(); }
	}

	if (strcmp(mode, "generate") != 0 && strcmp(mode, "chat") != 0) {
		fprintf(stderr, "unknown mode: %s\n", mode);
		error_usage();
	}

	Llama2cParam param;

	param.checkpoint_path = checkpoint_path;
	param.tokenizer_path = tokenizer_path;
	param.rng_seed = rng_seed;
	param.steps = steps;
	param.temperature = temperature;
	param.topp = topp;

	run(&param, mode, prompt, system_prompt);

	return 0;
}

#else

int main(int argc, char *argv[]) {

	Llama2cParam param;

	param.checkpoint_path = "app0:stories15M.bin";
	param.tokenizer_path = "app0:tokenizer.bin";
	param.rng_seed = time(NULL);
	param.steps = 256;
	param.temperature = 0.8f;
	param.topp = 0.9f;

	char *prompt = "";
	char *mode = "generate";
	char *system_prompt = "";

#ifdef PROMPT_FROM_FILE

	long length;
	FILE *fprompt = fopen("app0:prompt.txt", "rb");
	if (fprompt)
	{
		fseek(fprompt, 0, SEEK_END);
		length = ftell(fprompt);
		fseek(fprompt, 0, SEEK_SET);
		prompt = malloc(length);
		if (prompt)
		{
			fread(prompt, 1, length, fprompt);
		}
		else
		{
			prompt = "Failed to allocate memory for the prompt";
		}
		fclose(fprompt);
	}
	else
	{
		prompt = "Failed to open app0:prompt.txt";
	}

#else

	prompt = "Story about testing";

#endif

	fprintf(stderr, "\nPrompt: %s\n\n", prompt);

	run(&param, mode, prompt, system_prompt);

#ifdef PROMPT_FROM_FILE

	free(prompt);

#endif

	return 0;
}

#endif
