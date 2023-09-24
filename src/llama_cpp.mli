open Bigarray

type token = int32

type file_type =
  | ALL_F32
  | MOSTLY_F16
  | MOSTLY_Q4_0
  | MOSTLY_Q4_1
  | MOSTLY_Q4_1_SOME_F16
  | MOSTLY_Q8_0
  | MOSTLY_Q5_0
  | MOSTLY_Q5_1
  | MOSTLY_Q2_K
  | MOSTLY_Q3_K_S
  | MOSTLY_Q3_K_M
  | MOSTLY_Q3_K_L
  | MOSTLY_Q4_K_S
  | MOSTLY_Q4_K_M
  | MOSTLY_Q5_K_S
  | MOSTLY_Q5_K_M
  | MOSTLY_Q6_K
  | GUESSED

type vocab_type = Spm (** Sentencepiece *) | Bpe (** Byte Pair Encoding *)

type logits = (float, float32_elt, c_layout) Array2.t

type embeddings = (float, float32_elt, c_layout) Array1.t

type token_type =
  | Undefined
  | Normal
  | Unknown
  | Control
  | User_defined
  | Unused
  | Byte

val zero_token : int32

module Token_buffer : sig
  (** The type of token buffers, represented as arrays of {!int32}. *)
  type t = (token, int32_elt, c_layout) Array1.t

  (** [dim arr] is the length of the token buffer. *)
  val dim : t -> int

  (** [init len f] initializes a token buffer of length [len] with elements [f 0] to [f (len - 1)].
      This is an alias to {!Array1.init}. *)
  val init : int -> (int -> token) -> (token, int32_elt, c_layout) Array1.t

  (** [sub] is an alias to {!Array1.sub} *)
  val sub : t -> int -> int -> t

  (** [iter f arr] iterates over [arr] using [f]. *)
  val iter : (token -> unit) -> t -> unit

  (** [iteri f arr] iterates over [arr] using [f]. *)
  val iteri : (int -> token -> unit) -> t -> unit

  (** [to_seq arr] constructs a sequence over the array [arr]. Note that [arr] is not copied. *)
  val to_seq : t -> token Seq.t

  (** [of_seq seq] constructs an array from a sequence. *)
  val of_seq : token Seq.t -> t

  (** [of_array] is an alias to {!Array1.of_array}. *)
  val of_array : int32 array -> t

  (** [of_list ls] is [of_array (Array.of_list ls)]. *)
  val of_list : int32 list -> t
end

module Log_level : sig
  type t = Error | Warn | Info
end

module Context_params :
sig

  type t

  val make :
    seed:int ->
    n_ctx:int32 ->
    n_batch:int32 ->
    n_gpu_layers:int32 ->
    main_gpu:int32 ->
    tensor_split:float array ->
    rope_freq_base:float ->
    rope_freq_scale:float ->
    progress_callback:(float -> unit) ->
    low_vram:bool ->
    mul_mat_q:bool ->
    f16_kv:bool ->
    logits_all:bool ->
    vocab_only:bool -> use_mmap:bool -> use_mlock:bool -> embedding:bool -> t

  val default : unit -> t

  val seed : t -> int

  val n_ctx : t -> int32

  val n_batch : t -> int32

  val n_gpu_layers : t -> int32

  val main_gpu : t -> int32

  val tensor_split : t -> float array

  val rope_freq_base : t -> float

  val rope_freq_scale : t -> float

  val low_vram : t -> bool

  val mul_mat_q : t -> bool

  val f16_kv : t -> bool

  val logits_all : t -> bool

  val vocab_only : t -> bool

  val use_mmap : t -> bool

  val use_mlock : t -> bool

  val embedding : t -> bool
end

module Model_quantize_params :
sig

  type t

  val default : unit -> t

  val nthread : t -> int

  val ftype : t -> file_type

  val allow_requantize : t -> bool

  val quantize_output_tensor : t -> bool

  val only_copy : t -> bool
end

module Token_data_array : sig
  type t

  type logits = (float, float32_elt, c_layout) Array1.t

  val create : logits -> t

  val write_logits : t -> logits -> unit
end

module Grammar_element :
sig
  type gretype =
  | END (** End of rule definition *)
  | ALT (** Start of alternate definition for rule *)
  | RULE_REF (** Non-terminal element: reference to rule *)
  | CHAR (** Terminal element: character (code point) *)
  | CHAR_NOT (** Inverse char(s) (\[^a\], \[^a-b\] \[^abc\]) *)
  | CHAR_RNG_UPPER (** Modifies a preceding [CHAR] or [CHAR_ALT] to be
                       an inclusive range (\[a-z\]) *)
  | CHAR_ALT (** Modifies a preceding [CHAR] or [CHAR_ALT] to
                 add an alternate char to match (\[ab\], \[a-zA\])*)

  type t = {
    type_ : gretype ;
    value : int (** Unicode code point or rule ID *)
  }
end

module Timings :
sig
  type t = {
    t_start_ms : float ;
    t_end_ms : float ;
    t_load_ms : float ;
    t_sample_ms : float ;
    t_p_eval_ms : float ;
    t_eval_ms : float ;
    n_sample : int32 ;
    n_p_eval : int32 ;
    n_eval : int32
  }
end

type model

type context

type grammar

(** Initialize the llama + ggml backend
    If numa is true, use NUMA optimizations
    Call once at the start of the program *)
val backend_init : numa:bool -> unit

(** Call once at the end of the program - currently only used for MPI *)
val backend_free : unit -> unit

val load_model_from_file :
  string ->
  Context_params.t ->
  model option

(* TODO: should we use a finalizer to free those? *)
val free_model : model -> unit

val new_context_with_model :
  model ->
  Context_params.t ->
  context

(* TODO: should we use a finalizer to free those? *)
val free : context -> unit

val time_us : unit -> int64

val max_devices : unit -> unit

val mmap_supported : unit -> bool

val mlock_supported : unit -> bool

val n_vocab : context -> int

val n_ctx : context -> int

val n_ctx_train : context -> int

val n_embd : context -> int

val vocab_type : context -> vocab_type

val model_n_vocab : model -> int

val model_n_ctx : model -> int

val model_n_ctx_train : model -> int

val model_n_embd : model -> int

(** Get a string describing the model type *)
val model_desc : model -> string

(** Returns the total size of all the tensors in the model in bytes *)
val model_size : model -> int

(** Returns the total number of parameters in the model *)
val model_n_params : model -> int

(** Returns [true] on success *)
val model_quantize : fname_inp:string -> fname_out:string -> Model_quantize_params.t -> bool

(** Apply a LoRA adapter to a loaded model
    path_base_model is the path to a higher quality model to use as a base for
    the layers modified by the adapter. Can be NULL to use the current loaded model.
    The model needs to be reloaded before applying a new adapter, otherwise the adapter
    will be applied on top of the previous one
    Returns [true] on success *)
val model_apply_lora_from_file : model -> path_lora:string -> path_base_model:string -> n_threads:int -> bool

(** Returns the number of tokens in the KV cache *)
val get_kv_cache_token_count : context -> int

(** Sets the current rng seed. *)
val set_rng_seed : context -> int -> unit


 (** Returns the maximum size in bytes of the state (rng, logits, embedding
     and kv_cache) - will often be smaller after compacting tokens *)
val get_state_size : context -> int

(** Copies the state to the specified destination address.
    Destination needs to have allocated enough memory.
    Returns the number of bytes copied *)
val copy_state_data : context ->
  (char, int8_unsigned_elt, c_layout) Array1.t -> int

(** Set the state reading from the specified address
    Returns the number of bytes read *)
val set_state_data : context ->
  (char, int8_unsigned_elt, c_layout) Array1.t -> int

(** Clones a context. The model and parameters are not cloned however, and must be given as arguments. *)
val clone : context -> model -> Context_params.t -> context

(** Load session file in given buffer. If buffer is too small, returns [None], otherwise returns
    the number of elements actually written in the buffer. *)
val load_session_file : context -> path_session:string -> Token_buffer.t -> int option

(** Save session file in given buffer. Return [true] on success. *)
val save_session_file : context -> path_session:string -> Token_buffer.t -> bool

(** [eval ctx tbuff ~n_tokens ~n_past ~n_threads] runs the llama inference to obtain the logits and probabilities for the next token.
    The prefix of [tbuff] of length [n_tokens] is the provided batch of new tokens to process.
    [n_past] is the number of tokens to use from previous [eval] calls.
    Returns [Some logits] on success where [logits] is a two-dimensional array with number of rows equal to [n_tokens] and
    number of colums equal to [n_vocab]. The logits for the last token are stored in the last row.
    These logits can be mutated in order to change the probabilities of the next token.
 *)
val eval : context -> Token_buffer.t -> n_tokens:int -> n_past:int -> n_threads:int -> logits option

(** Same as llama_eval, but use a float matrix input directly. *)
val eval_embd : context -> (float, float32_elt, c_layout) Array1.t -> n_tokens:int -> n_past:int -> n_threads:int -> logits option

(** Export a static computation graph for context of 511 and batch size of 1
    NOTE: since this functionality is mostly for debugging and demonstration purposes, we hardcode these
          parameters here to keep things simple
    IMPORTANT: do not use for anything else other than debugging and testing! *)
val eval_export : context -> string -> bool

(** Get the embeddings for the input shape: [n_embd] (1-dimensional) *)
val get_embeddings : context -> embeddings

(** Vocab *)

val token_get_text : context -> token -> string

val token_get_score : context -> token -> float

val token_get_type : context -> token -> token_type

(** Special tokens *)

(** beginning-of-sentence *)
val token_bos : context -> token

(** end-of-sentence *)
val token_eos : context -> token

(** next-line *)
val token_nl : context -> token

(** Tokenization *)

(** Convert the provided text into tokens.
    The tokens buffer must be large enough to hold the resulting tokens.
    Returns the number of written tokens on success, no more than n_max_tokens
    Returns [Error (`Too_many_tokens n)] on failure - the number of tokens that would have been returned *)
val tokenize : context -> text:string -> Token_buffer.t -> n_max_tokens:int -> add_bos:bool -> (int, [`Too_many_tokens of int]) result

val tokenize_with_model : model -> text:string -> Token_buffer.t -> n_max_tokens:int -> add_bos:bool -> (int, [`Too_many_tokens of int]) result

(** Token Id -> Piece.
    Uses the vocabulary in the provided context.
    Does not write null terminator to the buffer.
    User code is responsible to remove the leading whitespace of the first non-BOS token when decoding multiple tokens. *)
val token_to_piece : context -> token -> (string, [`Invalid_token]) result

val token_to_piece_with_model : model -> token -> (string, [`Invalid_token]) result

(** Grammar *)

val grammar_init : Grammar_element.t array array -> start_rule_index:int -> grammar

val grammar_copy : grammar -> grammar

(** Sampling functions *)

(** Repetition penalty described in CTRL academic paper https://arxiv.org/abs/1909.05858, with negative logit fix. *)
val sample_repetition_penalty : context -> candidates:Token_data_array.t -> last_tokens:Token_buffer.t -> penalty:float -> unit

(** Frequency and presence penalties described in OpenAI API https://platform.openai.com/docs/api-reference/parameter-details. *)
val sample_frequency_and_presence_penalties : context -> candidates:Token_data_array.t -> last_tokens:Token_buffer.t -> alpha_frequency:float ->  alpha_presence:float -> unit

(** Apply classifier-free guidance to the logits as described in academic paper "Stay on topic with Classifier-Free Guidance" https://arxiv.org/abs/2306.17806
    @param candidates A vector of `llama_token_data` containing the candidate tokens, the logits must be directly extracted from the original generation context without being sorted.
    @param guidance_ctx A separate context from the same model. Other than a negative prompt at the beginning, it should have all generated and user input tokens copied from the main context.
    @param scale Guidance strength. 1.0f means no guidance. Higher values mean stronger guidance. *)
val sample_classifier_free_guidance : context -> candidates:Token_data_array.t -> guidance_ctx:context -> scale:float -> unit

(** Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits. *)
val sample_softmax : context -> candidates:Token_data_array.t -> unit

(** Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751 *)
val sample_top_k : context -> candidates:Token_data_array.t -> k:int -> min_keep:int -> unit

(** Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751 *)
val sample_top_p : context -> candidates:Token_data_array.t -> p:float -> min_keep:int -> unit

(** Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/. *)
val sample_tail_free : context -> candidates:Token_data_array.t -> z:float -> min_keep:int -> unit

(** Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666. *)
val sample_typical : context -> candidates:Token_data_array.t -> p:float -> min_keep:int -> unit

val sample_temperature : context -> candidates:Token_data_array.t -> temp:float -> unit

(** Apply constraints from grammar *)
val sample_grammar : context -> candidates:Token_data_array.t -> grammar -> unit

(** Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
    @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal. *)
val sample_token_mirostat : context -> candidates:Token_data_array.t -> tau:float -> eta:float -> m:int -> float * token


(** Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal. *)
val sample_token_mirostat_v2 : context -> candidates:Token_data_array.t -> tau:float -> eta:float -> float * token

(** Selects the token with the highest probability. *)
val sample_token_greedy : context -> candidates:Token_data_array.t -> token

(** Randomly selects a token from the candidates based on their probabilities. *)
val sample_token : context -> candidates:Token_data_array.t -> token

(** Accepts the sampled token into the grammar *)
val grammar_accept_token : context -> grammar -> token -> unit

(** Beam search *)

type beam_view = {
  tokens : Token_buffer.t ;
  p : float ;
  eob : bool
}

type beam_search_callback =
  beam_views:beam_view array ->
  common_prefix_length:int ->
  last_call:bool ->
  unit

val beam_search : context -> beam_search_callback -> n_beams:int -> n_past:int -> n_predict:int -> n_threads:int -> unit

(** Performance information *)

val get_timings : context -> Timings.t

val print_timings : context -> unit

val reset_timings : context -> unit

(** Print system information *)

val print_system_info : unit -> string

val log_set : (Log_level.t -> string -> unit) -> unit
