open Ctypes

module Types = Llama_fixed_types.Types
module Stubs = Llama_generated

module Internal = struct

  module Context_params =
  struct
    let make
        ~seed ~n_ctx ~n_batch ~n_gpu_layers ~main_gpu
        ~tensor_split ~rope_freq_base ~rope_freq_scale
        ~progress_callback ~progress_callback_user_data
        ~low_vram ~mul_mat_q ~f16_kv ~logits_all ~vocab_only ~use_mmap ~use_mlock ~embedding =
      let open Types.Context_params in
      let result = make repr in
      setf result Fields.seed seed ;
      setf result Fields.n_ctx n_ctx ;
      setf result Fields.n_batch n_batch ;
      setf result Fields.n_gpu_layers n_gpu_layers ;
      setf result Fields.main_gpu main_gpu ;

      setf result Fields.tensor_split tensor_split ;

      setf result Fields.rope_freq_base rope_freq_base ;
      setf result Fields.rope_freq_scale rope_freq_scale ;

      setf result Fields.progress_callback progress_callback ;
      setf result Fields.progress_callback_user_data progress_callback_user_data ;

      setf result Fields.low_vram low_vram ;
      setf result Fields.mul_mat_q mul_mat_q ;
      setf result Fields.f16_kv f16_kv ;
      setf result Fields.logits_all logits_all ;
      setf result Fields.vocab_only vocab_only ;
      setf result Fields.use_mmap use_mmap ;
      setf result Fields.use_mlock use_mlock ;
      setf result Fields.embedding embedding ;
      result
  end

end

type context_params = (Types.Context_params.t, [ `Struct ]) structured

let funptr_of_function fn f =
  coerce (Foreign.funptr fn) (static_funptr fn) f

let context_params
    ~seed
    ~n_ctx
    ~n_batch
    ~n_gpu_layers
    ~main_gpu
    ~tensor_split
    ~rope_freq_base
    ~rope_freq_scale
    ~(progress_callback:(float -> unit))
    ~low_vram
    ~mul_mat_q
    ~f16_kv
    ~logits_all
    ~vocab_only
    ~use_mmap
    ~use_mlock
    ~embedding
  : context_params =
  let tensor_split = Bigarray.Array1.of_array Bigarray.Float32 Bigarray.c_layout tensor_split in
  let tensor_split = Ctypes.bigarray_start Ctypes.array1 tensor_split in
  let progress_callback =
    (funptr_of_function (float @-> (ptr void) @-> returning void) (fun flt _ -> progress_callback flt))
  in
  Internal.Context_params.make
    ~seed:(Unsigned.UInt32.of_int seed)
    ~n_ctx
    ~n_batch
    ~n_gpu_layers
    ~main_gpu
    ~tensor_split
    ~rope_freq_base
    ~rope_freq_scale
    ~progress_callback
    ~progress_callback_user_data:Ctypes.null
    ~low_vram
    ~mul_mat_q
    ~f16_kv
    ~logits_all
    ~vocab_only
    ~use_mmap
    ~use_mlock
    ~embedding
