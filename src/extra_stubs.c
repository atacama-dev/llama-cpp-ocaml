#include "ctypes_cstubs_internals.h"
#include "llama.h"

void init_token_data_array(llama_token_data_array* array)
{
     struct llama_token_data* data = array->data;

     for (int i = 0; i < array->size; i++) {
          data[i].id = i;
          data[i].logit = 0.0f;
          data[i].p = 0.0f;
     }
}

void write_logits(llama_token_data_array* array, float* logits)
{
     const int dim = array->size;

     for (int i = 0; i < dim; i++) {
          array->data[i].logit = logits[i];
     }
}
