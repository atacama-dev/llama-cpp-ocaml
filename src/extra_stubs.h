#ifndef EXTRA_STUBS_H
#define EXTRA_STUBS_H

#include "llama.h"

void init_token_data_array(llama_token_data_array* array);
void write_logits(llama_token_data_array* array, float* logits);

#endif // EXTRA_STUBS_H
