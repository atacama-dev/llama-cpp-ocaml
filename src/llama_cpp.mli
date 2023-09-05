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

type vocab_type = Spm | Bpe

type token_buff = (int32, int32_elt, c_layout) Array1.t

type logits = (float, float32_elt, c_layout) Array2.t

type embeddings = (float, float32_elt, c_layout) Array1.t


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
end

module Token_data :
sig

  type t  = {
    id : token;
    logit : float;
    p: float
  }

end

module Token_data_array : sig
  type t

  val sorted : t -> bool

  val get : t -> int -> Token_data.t

  val set : t -> int -> Token_data.t -> unit
end

type model

type context

(** Initialize the llama + ggml backend
    If numa is true, use NUMA optimizations
    Call once at the start of the program *)
val backend_init : numa:bool -> unit

(** Call once at the end of the program - currently only used for MPI *)
val backend_free : unit -> unit

val load_model_from_file :
  string ->
  Context_params.t ->
  model

val free_model : model -> unit

val new_context_with_model :
  model ->
  Context_params.t ->
  context

val free : context -> unit

val time_us : unit -> int64

val max_devices : unit -> unit

val mmap_supported : unit -> bool

val mlock_supported : unit -> bool

val n_vocab : context -> int

val n_ctx : context -> int

val n_embd : context -> int

val vocab_type : context -> vocab_type

val model_n_vocab : model -> int

val model_n_ctx : model -> int

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

(** Load session file in given buffer. If buffer is too small, returns [None], otherwise returns
    the number of elements actually written in the buffer. *)
val load_session_file : context -> path_session:string -> token_buff -> int option

(** Save session file in given buffer. Return [true] on success. *)
val save_session_file : context -> path_session:string -> token_buff -> bool

(** Run the llama inference to obtain the logits and probabilities for the next token.
    tokens + n_tokens is the provided batch of new tokens to process
    n_past is the number of tokens to use from previous eval calls
    Returns [Some logits] on success.

    The logits for the last token are stored in the last row
    Can be mutated in order to change the probabilities of the next token
    Rows: n_tokens
    Cols: n_vocab
 *)
val eval : context -> token_buff -> n_tokens:int -> n_past:int -> n_threads:int -> logits option

(** Same as llama_eval, but use float matrix input directly. *)
val eval_embd : context -> (float, float32_elt, c_layout) Array1.t -> n_tokens:int -> n_past:int -> n_threads:int -> logits option

(** Export a static computation graph for context of 511 and batch size of 1
    NOTE: since this functionality is mostly for debugging and demonstration purposes, we hardcode these
          parameters here to keep things simple
    IMPORTANT: do not use for anything else other than debugging and testing! *)
val eval_export : context -> string -> bool

(** Get the embeddings for the input
    shape: [n_embd] (1-dimensional) *)
val get_embeddings : context -> embeddings

(** Vocab *)

val token_get_text : context -> token -> string

val token_get_score : context -> token -> float
