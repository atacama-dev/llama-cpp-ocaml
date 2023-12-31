open Llama_fixed_types.Types

module Make = functor (T : Ctypes.FOREIGN) -> struct
  open Ctypes
  open T

  let model_default_params =
    foreign "llama_model_default_params" (void @-> returning Model_params.repr)

  let context_default_params =
    foreign "llama_context_default_params" (void @-> returning Context_params.repr)

  let model_quantize_default_params =
    foreign "llama_model_quantize_default_params" (void @-> returning Model_quantize_params.repr)

  let backend_init =
    foreign "llama_backend_init" (bool @-> returning void)

  let backend_free =
    foreign "llama_backend_free" (void @-> returning void)

  let load_model_from_file =
    foreign "llama_load_model_from_file" (string @-> Model_params.repr @-> returning (ptr Model.repr))

  let free_model =
    foreign "llama_free_model" (ptr Model.repr @-> returning void)

  let new_context_with_model =
    foreign "llama_new_context_with_model" (ptr Model.repr @-> Context_params.repr @-> returning (ptr Context.repr))

  let free =
    foreign "llama_free" (ptr Context.repr @-> returning void)

  let time_us =
    foreign "llama_time_us" (void @-> returning int64_t)

  let max_devices =
    foreign "llama_max_devices" (void @-> returning void)

  let mmap_supported =
    foreign "llama_mmap_supported" (void @-> returning bool)

  let mlock_supported =
    foreign "llama_mlock_supported" (void @-> returning bool)

  let get_model =
    foreign "llama_get_model" (ptr Context.repr @-> returning (ptr (Model.repr)))

  let n_ctx =
    foreign "llama_n_ctx" (ptr Context.repr @-> returning int)

  let n_embd =
    foreign "llama_n_embd" (ptr Model.repr @-> returning int)

  let vocab_type =
    foreign "llama_vocab_type" (ptr Model.repr @-> returning Vocab_type.repr)

  let n_vocab =
    foreign "llama_n_vocab" (ptr Model.repr @-> returning int)

  let n_ctx_train =
    foreign "llama_n_ctx_train" (ptr Model.repr @-> returning int)

  (* Get a string describing the model type *)
  let model_desc =
    foreign "llama_model_desc" (ptr Model.repr @-> ptr char @-> size_t @-> returning int)

  (* Returns the total size of all the tensors in the model in bytes *)
  let model_size =
    foreign "llama_model_size" (ptr Model.repr @-> returning uint64_t)

  (* Returns the total number of parameters in the model *)
  let model_n_params =
    foreign "llama_model_n_params" (ptr Model.repr @-> returning uint64_t)

  (* TODO:
      LLAMA_API struct ggml_tensor * llama_get_model_tensor(struct llama_model * model, const char * name);
  *)

  (* Returns 0 on success *)
  let model_quantize =
    foreign "llama_model_quantize" (ptr char @-> ptr char @-> ptr Model_quantize_params.repr @-> returning int)

  (* Apply a LoRA adapter to a loaded model *)
  (* path_base_model is the path to a higher quality model to use as a base for *)
  (* the layers modified by the adapter. Can be NULL to use the current loaded model. *)
  (* The model needs to be reloaded before applying a new adapter, otherwise the adapter *)
  (* will be applied on top of the previous one *)
  (* Returns 0 on success *)
  let model_apply_lora_from_file =
    foreign "llama_model_apply_lora_from_file" (ptr Model.repr @-> ptr char @-> float @-> ptr char @-> int @-> returning int)

  (* Returns the number of tokens in the KV cache *)
  (* DEPRECATED *)
  (* let get_kv_cache_token_count = *)
  (*   foreign "llama_get_kv_cache_token_count" (ptr Context.repr @-> returning int) *)

  let kv_cache_tokens_rm =
    foreign "llama_kv_cache_tokens_rm" (ptr Context.repr @-> int32_t @-> int32_t @-> returning void)

  let kv_cache_seq_rm =
    foreign "llama_kv_cache_seq_rm" (ptr Context.repr @-> Seq_id.repr @-> int32_t @-> int32_t @-> returning void)

  let kv_cache_seq_cp =
    foreign "llama_kv_cache_seq_cp"
      (ptr Context.repr @-> Seq_id.repr @-> Seq_id.repr @-> int32_t @-> int32_t @-> returning void)

  let kv_cache_seq_keep =
    foreign "llama_kv_cache_seq_keep" (ptr Context.repr @-> Seq_id.repr @-> returning void)

  let kv_cache_seq_shift =
    foreign "llama_kv_cache_seq_shift"
      (ptr Context.repr @-> Seq_id.repr @-> Pos.repr @-> Pos.repr @-> Pos.repr @-> returning void)

  (* Returns the maximum size in bytes of the state (rng, logits, embedding *)
  (* and kv_cache) - will often be smaller after compacting tokens *)
  let get_state_size =
    foreign "llama_get_state_size" (ptr Context.repr @-> returning size_t)

  (* Copies the state to the specified destination address. *)
  (* Destination needs to have allocated enough memory. *)
  (* Returns the number of bytes copied *)
  let copy_state_data =
    foreign "llama_copy_state_data" (ptr Context.repr @-> ptr uint8_t @-> returning size_t)

  (* Set the state reading from the specified address *)
  (* Returns the number of bytes read *)
  let set_state_data =
    foreign "llama_set_state_data" (ptr Context.repr @-> ptr uint8_t @-> returning size_t)

  let load_session_file =
    foreign "llama_load_session_file" (ptr Context.repr @-> ptr char @-> ptr Token.repr @-> size_t @-> ptr size_t @-> returning bool)

  let save_session_file =
    foreign "llama_save_session_file" (ptr Context.repr @-> ptr char @-> ptr Token.repr @-> size_t @-> returning bool)

  (* Run the llama inference to obtain the logits and probabilities for the next token. *)
  (* tokens + n_tokens is the provided batch of new tokens to process *)
  (* n_past is the number of tokens to use from previous eval calls *)
  (* Returns 0 on success *)
  (* DEPRECATED *)
  (* let eval = *)
  (*   foreign "llama_eval" *)
  (*     (ptr Context.repr @-> ptr Token.repr @-> int @-> int @-> int @-> returning int) *)

  (* Same as llama_eval, but use float matrix input directly. *)
  (* DEPRECATED *)
  (* let eval_embd = *)
  (*   foreign "llama_eval_embd" *)
  (*     (ptr Context.repr @-> ptr float @-> int @-> int @-> int @-> returning int) *)

  (* Export a static computation graph for context of 511 and batch size of 1 *)
  (* NOTE: since this functionality is mostly for debugging and demonstration purposes, we hardcode these *)
  (*       parameters here to keep things simple *)
  (* IMPORTANT: do not use for anything else other than debugging and testing! *)
  (* DEPRECATED *)
  (* let eval_export = *)
  (*   foreign "llama_eval_export" (ptr Context.repr @-> ptr char @-> returning int) *)

  let batch_get_one =
    foreign "llama_batch_get_one" (ptr Token.repr @-> int32_t @-> Pos.repr @-> Seq_id.repr @-> returning Batch.repr)

  let batch_init =
    foreign "llama_batch_init" (int32_t @-> int32_t @-> returning Batch.repr)

  let batch_free =
    foreign "llama_batch_free" (Batch.repr @-> returning void)

  let decode =
    foreign "llama_decode" (ptr Context.repr @-> Batch.repr @-> returning int)

  let set_n_threads =
    foreign "llama_set_n_threads" (ptr Context.repr @-> uint32_t @-> uint32_t @-> returning void)


  (* Token logits obtained from the last call to llama_eval() *)
  (* The logits for the last token are stored in the last row *)
  (* Can be mutated in order to change the probabilities of the next token *)
  (* Rows: n_tokens *)
  (* Cols: n_vocab *)
  let get_logits =
    foreign "llama_get_logits" (ptr Context.repr @-> returning (ptr float))

  (* Get the embeddings for the input *)
  (* shape: [n_embd] (1-dimensional) *)
  let get_embeddings =
    foreign "llama_get_embeddings" (ptr Context.repr @-> returning (ptr float))

  (* Vocab *)

  let token_get_text =
    foreign "llama_token_get_text" (ptr Context.repr @-> Token.repr @-> returning (ptr char))

  let token_get_score =
    foreign "llama_token_get_score" (ptr Context.repr @-> Token.repr @-> returning float)

  let token_get_type =
    foreign "llama_token_get_type" (ptr Context.repr @-> Token.repr @-> returning Token_type.repr)

  (* beginning-of-sentence *)
  let token_bos =
    foreign "llama_token_bos" (ptr Context.repr @-> returning Token.repr)

  (* end-of-sentence *)
  let token_eos =
    foreign "llama_token_eos" (ptr Context.repr @-> returning Token.repr)

  (* next-line *)
  let token_nl =
    foreign "llama_token_nl" (ptr Context.repr @-> returning Token.repr)

  (* Tokenization *)

  (* Convert the provided text into tokens. *)
  (* The tokens pointer must be large enough to hold the resulting tokens. *)
  (* Returns the number of tokens on success, no more than n_max_tokens *)
  (* Returns a negative number on failure - the number of tokens that would have been returned *)
  let tokenize =
    foreign "llama_tokenize" (ptr Model.repr @-> ptr char @-> int @-> ptr Token.repr @-> int @-> bool @-> returning int)

  (* Token Id -> Piece. *)
  (* Uses the vocabulary in the provided context. *)
  (* Does not write null terminator to the buffer. *)
  (* User code is responsible to remove the leading whitespace of the first non-BOS token when decoding multiple tokens. *)
  let token_to_piece =
    foreign "llama_token_to_piece" (ptr Model.repr @-> Token.repr @-> ptr char @-> int @-> returning int)

  (* Grammar *)

  let grammar_init =
    foreign "llama_grammar_init"
      (ptr (ptr Grammar_element.repr) @-> size_t @-> size_t @-> returning (ptr Grammar.repr))

  let grammar_free =
    foreign "llama_grammar_free" (ptr Grammar.repr @-> returning void)

  let grammar_copy =
    foreign "llama_grammar_copy" (ptr Grammar.repr @-> returning (ptr Grammar.repr))

  (* Sampling functions *)

  (* Sets the current rng seed. *)
  let set_rng_seed =
    foreign "llama_set_rng_seed" (ptr Context.repr @-> uint32_t @-> returning void)

  (* Repetition penalty described in CTRL academic paper https://arxiv.org/abs/1909.05858, with negative logit fix. *)
  let sample_repetition_penalty =
    foreign "llama_sample_repetition_penalty"
      (ptr Context.repr @-> ptr Token_data_array.repr @-> ptr Token.repr @-> size_t @-> float @-> returning void)

  (* Frequency and presence penalties described in OpenAI API https://platform.openai.com/docs/api-reference/parameter-details. *)
  let sample_frequency_and_presence_penalties =
    foreign "llama_sample_frequency_and_presence_penalties"
      (ptr Context.repr @-> ptr Token_data_array.repr @-> ptr Token.repr @-> size_t @-> float @-> float @-> returning void)

     (* @details Apply classifier-free guidance to the logits as described in academic paper "Stay on topic with Classifier-Free Guidance" https://arxiv.org/abs/2306.17806 *)
     (* @param candidates A vector of `llama_token_data` containing the candidate tokens, the logits must be directly extracted from the original generation context without being sorted. *)
     (* @params guidance_ctx A separate context from the same model. Other than a negative prompt at the beginning, it should have all generated and user input tokens copied from the main context. *)
     (* @params scale Guidance strength. 1.0f means no guidance. Higher values mean stronger guidance. *)
  let sample_classifier_free_guidance =
    foreign "llama_sample_classifier_free_guidance"
      (ptr Context.repr @->
       ptr Token_data_array.repr @->
       ptr Context.repr @->
       float @->
       returning void)

  (* Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits. *)
  let sample_softmax =
    foreign "llama_sample_softmax" (ptr Context.repr @-> ptr Token_data_array.repr @-> returning void)

  (* Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751 *)
  let sample_top_k =
    foreign "llama_sample_top_k" (ptr Context.repr @-> ptr Token_data_array.repr @-> int @-> size_t @-> returning void)

  (* Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751 *)
  let sample_top_p =
    foreign "llama_sample_top_p" (ptr Context.repr @-> ptr Token_data_array.repr @-> float @-> size_t @-> returning void)

  (* Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/. *)
  let sample_tail_free =
    foreign "llama_sample_tail_free" (ptr Context.repr @-> ptr Token_data_array.repr @-> float @-> size_t @-> returning void)

  (* Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666. *)
  let sample_typical =
    foreign "llama_sample_typical" (ptr Context.repr @-> ptr Token_data_array.repr @-> float @-> size_t @-> returning void)

  let sample_temp =
    foreign "llama_sample_temp" (ptr Context.repr @-> ptr Token_data_array.repr @-> float @-> returning void)

  (* DEPRECATED *)
  (* let sample_temperature = *)
  (*   foreign "llama_sample_temperature" (ptr Context.repr @-> ptr Token_data_array.repr @-> float @-> returning void) *)

  (* Apply constraints from grammar *)
  let sample_grammar =
    foreign "llama_sample_grammar" (ptr Context.repr @-> ptr Token_data_array.repr @-> ptr Grammar.repr @-> returning void)

   (* @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words. *)
   (* @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text. *)
   (* @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text. *)
   (* @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates. *)
   (* @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm. *)
   (* @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal. *)
  let sample_token_mirostat =
    foreign "llama_sample_token_mirostat" (ptr Context.repr @-> ptr Token_data_array.repr @-> float @-> float @-> int @-> ptr float @-> returning Token.repr)

     (* @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words. *)
     (* @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text. *)
     (* @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text. *)
     (* @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates. *)
     (* @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal. *)
  let sample_token_mirostat_v2 =
    foreign "llama_sample_token_mirostat_v2" (ptr Context.repr @-> ptr Token_data_array.repr @-> float @-> float @-> ptr float @-> returning Token.repr)

  (* @details Selects the token with the highest probability. *)
  let sample_token_greedy =
    foreign "llama_sample_token_greedy" (ptr Context.repr @-> ptr Token_data_array.repr @-> returning Token.repr)

  (* @details Randomly selects a token from the candidates based on their probabilities. *)
  let sample_token =
    foreign "llama_sample_token" (ptr Context.repr @-> ptr Token_data_array.repr @-> returning Token.repr)

  (* @details Accepts the sampled token into the grammar *)
  let grammar_accept_token =
    foreign "llama_grammar_accept_token" (ptr Context.repr @-> ptr Grammar.repr @-> Token.repr @-> returning void)

  (* Beam search *)

  (* @details Deterministically returns entire sentence constructed by a beam search. *)
  (* @param ctx Pointer to the llama_context. *)
  (* @param callback Invoked for each iteration of the beam_search loop, passing in beams_state. *)
  (* @param callback_data A pointer that is simply passed back to callback. *)
  (* @param n_beams Number of beams to use. *)
  (* @param n_past Number of tokens already evaluated. *)
  (* @param n_predict Maximum number of tokens to predict. EOS may occur earlier. *)
  let beam_search =
    foreign "llama_beam_search"
      (ptr Context.repr @-> (Foreign.funptr Ctypes.(ptr void @-> Beams_state.repr @-> returning void)) @-> ptr void @-> size_t @-> int @-> int @-> returning void)


  let get_timings =
    foreign "llama_get_timings"
      (ptr Context.repr @-> returning Timings.repr)

  let print_timings =
    foreign "llama_print_timings"
      (ptr Context.repr @-> returning void)

  let reset_timings =
    foreign "llama_reset_timings"
      (ptr Context.repr @-> returning void)

  let print_system_info =
    foreign "llama_print_system_info" (void @-> returning (ptr char))

  (* Set callback for all future logging events. *)
  (* If this is not called, or NULL is supplied, everything is output on stderr. *)
  let log_set =
    foreign "llama_log_set" ((Foreign.funptr Ctypes.(Log_level.repr @-> ptr char @-> ptr void @-> returning void)) @-> ptr void @-> returning void)

  let strlen =
    foreign "strlen" (ptr char @-> returning size_t)

  let init_token_data_array =
    foreign "init_token_data_array" (ptr Token_data_array.repr @-> returning void)

  let write_logits =
    foreign "write_logits" (ptr Token_data_array.repr @-> ptr float @-> returning void)
end
