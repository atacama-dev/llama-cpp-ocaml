type context_params

type model

type context

val context_params :
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
  vocab_only:bool ->
  use_mmap:bool ->
  use_mlock:bool ->
  embedding:bool ->
  context_params

val load_model_from_file :
  string ->
  context_params ->
  model

val free_model : model -> unit

val new_context_with_model :
  model ->
  context_params ->
  context

val free : context -> unit
