type token_id = int

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

module Token_data :
sig

  type t  = {
    id : token_id;
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
