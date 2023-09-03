open Ctypes

module Types = Llama_fixed_types.Types
module Stubs = Llama_functions.Make (Llama_generated)

let funptr_of_function fn f =
  coerce (Foreign.funptr fn) (static_funptr fn) f

type token_id = int

module Context_params =
struct

  type t = (Types.Context_params.t, [ `Struct ]) structured

  let make_internal
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

  let make
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
    : t =
    if Array.length tensor_split <> Types.max_devices then
      Format.kasprintf invalid_arg "Context_paramns.make: tensor_split length <> %d" Types.max_devices ;
    let tensor_split = Bigarray.Array1.of_array Bigarray.Float32 Bigarray.c_layout tensor_split in
    let tensor_split = Ctypes.bigarray_start Ctypes.array1 tensor_split in
    let progress_callback =
      (funptr_of_function (float @-> (ptr void) @-> returning void) (fun flt _ -> progress_callback flt))
    in
    make_internal
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

  let default = Stubs.context_default_params

  let seed cp = getf cp Types.Context_params.Fields.seed |> Unsigned.UInt32.to_int

  let n_ctx cp = getf cp Types.Context_params.Fields.n_ctx

  let n_batch cp = getf cp Types.Context_params.Fields.n_batch

  let n_gpu_layers cp = getf cp Types.Context_params.Fields.n_gpu_layers

  let main_gpu cp = getf cp Types.Context_params.Fields.main_gpu

  let tensor_split cp =
    let ptr = getf cp Types.Context_params.Fields.tensor_split in
    let arr = bigarray_of_ptr Ctypes.array1 Types.max_devices Bigarray.Float32 ptr in
    Array.init Types.max_devices (fun i -> arr.{i})

  let rope_freq_base cp = getf cp Types.Context_params.Fields.rope_freq_base

  let rope_freq_scale cp = getf cp Types.Context_params.Fields.rope_freq_scale

  let low_vram cp = getf cp Types.Context_params.Fields.low_vram

  let mul_mat_q cp = getf cp Types.Context_params.Fields.mul_mat_q

  let f16_kv cp = getf cp Types.Context_params.Fields.f16_kv

  let logits_all cp = getf cp Types.Context_params.Fields.logits_all

  let vocab_only cp = getf cp Types.Context_params.Fields.vocab_only

  let use_mmap cp = getf cp  Types.Context_params.Fields.use_mmap

  let use_mlock cp = getf cp  Types.Context_params.Fields.use_mlock

  let embedding cp = getf cp  Types.Context_params.Fields.embedding
end

module Token_data =
struct
  type t  = {
    id : token_id;
    logit : float;
    p: float
  }

  type internal = (Types.Token_data.t, [ `Struct ]) structured

  let to_internal ~id ~logit ~p =
    let open Types.Token_data in
    let result = make repr in
    setf result Fields.id id ;
    setf result Fields.logit logit ;
    setf result Fields.p p ;
    result

  let of_internal (td : internal) =
    let open Types.Token_data in
    let id = getf td Fields.id in
    let logit = getf td Fields.logit in
    let p = getf td Fields.p in
    { id; logit; p }
end

module Token_data_array =
struct
  type t = {
    data : Token_data.internal CArray.t ;
    sorted : bool
  }

  let sorted { sorted; data = _ } = sorted

  let get { data; _ } i = CArray.get data i |> Token_data.of_internal

  let set { data; _ } i { Token_data.id; logit; p } = CArray.set data i (Token_data.to_internal ~id ~logit ~p)
end

type model = Types.Model.t Ctypes_static.structure Ctypes_static.ptr

type context = Types.Context.t Ctypes_static.structure Ctypes_static.ptr

let load_model_from_file = Stubs.load_model_from_file

let free_model = Stubs.free_model

let new_context_with_model = Stubs.new_context_with_model

let free = Stubs.free
