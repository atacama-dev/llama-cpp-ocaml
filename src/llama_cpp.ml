open Bigarray
open Ctypes

module Types = Llama_fixed_types.Types
module Stubs = Llama_functions.Make (Llama_generated)

let funptr_of_function fn f =
  coerce (Foreign.funptr fn) (static_funptr fn) f

type token = int32

type file_type = Types.File_type.t =
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

type vocab_type = Types.Vocab_type.t = Spm | Bpe

type token_buff = (int32, int32_elt, c_layout) Array1.t

type logits = (float, float32_elt, c_layout) Array2.t

type embeddings = (float, float32_elt, c_layout) Array1.t

type token_type = Types.Token_type.t =
  | Undefined
  | Normal
  | Unknown
  | Control
  | User_defined
  | Unused
  | Byte

let zero_token = 0l

module Token_buffer = struct
  type t = (int32, int32_elt, c_layout) Array1.t

  let dim (arr : t) = Array1.dim arr

  let init f = Array1.init Int32 c_layout f

  let sub (arr : t) ofs len : t = Array1.sub arr ofs len

  let iter f (arr : t) =
    for i = 0 to Array1.dim arr - 1 do
      f (Array1.unsafe_get arr i)
    done

  let iteri f (arr : t) =
    for i = 0 to Array1.dim arr - 1 do
      f i (Array1.unsafe_get arr i)
    done

  let to_seq (arr : t) =
    let rec loop (i : int) () =
      if i >= dim arr then Seq.Nil
      else
        let elt = Array1.unsafe_get arr i in
        Seq.Cons (elt, loop (i + 1))
    in
    loop 0

  let of_seq (seq : int32 Seq.t) : t =
    Array.of_seq seq
    |> Array1.of_array Int32 c_layout

  let of_array (arr : int32 array) = Array1.of_array Int32 c_layout arr

  let of_list (ls : int32 list) = Array1.of_array Int32 c_layout (Array.of_list ls)
end


module Log_level =
struct
  type t = Types.Log_level.t = Error | Warn | Info
end

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
    let tensor_split = Array1.of_array Float32 c_layout tensor_split in
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
    let arr = bigarray_of_ptr Ctypes.array1 Types.max_devices Float32 ptr in
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

module Model_quantize_params =
struct
  type t = Types.Model_quantize_params.t structure ptr

  let default () =
    Ctypes.allocate Types.Model_quantize_params.repr
      (Stubs.model_quantize_default_params ())

  let nthread (qp : t) = !@ (qp |-> Types.Model_quantize_params.Fields.nthread)

  let ftype (qp : t) = !@ (qp |-> Types.Model_quantize_params.Fields.ftype)

  let allow_requantize (qp : t) = !@ (qp |-> Types.Model_quantize_params.Fields.allow_requantize)

  let quantize_output_tensor (qp : t) = !@ (qp |-> Types.Model_quantize_params.Fields.quantize_output_tensor)

  let only_copy (qp : t) = !@ (qp |-> Types.Model_quantize_params.Fields.only_copy)
end

module Token_data_array =
struct
  type t = Types.Token_data_array.t structure ptr

  type logits = (float, float32_elt, c_layout) Array1.t

  let make_internal len =
    let open Types.Token_data_array in
    let strct = make repr in
    let data = Ctypes.allocate_n Types.Token_data.repr ~count:len in
    setf strct Fields.data data ;
    setf strct Fields.size (Unsigned.Size_t.of_int len) ;
    setf strct Fields.sorted false ;
    let res = Ctypes.allocate repr strct in
    Stubs.init_token_data_array res ;
    res

  let write_logits tda logits =
    let dim = getf !@tda Types.Token_data_array.Fields.size |> Unsigned.Size_t.to_int in
    if dim <> Array1.dim logits then
      invalid_arg "write_logits: wrong dimensions" ;
    Stubs.write_logits tda (Ctypes.bigarray_start array1 logits)

  let create logits =
    let res = make_internal (Array1.dim logits) in
    write_logits res logits ;
    res
end

module Grammar_element =
struct
  type gretype = Types.Gretype.t =
  | END
  | ALT
  | RULE_REF
  | CHAR
  | CHAR_NOT
  | CHAR_RNG_UPPER
  | CHAR_ALT

  type t = {
    type_ : gretype ;
    value : int (* Unicode code point or rule ID *)
  }

  let make_internal { type_; value } =
    let open Types.Grammar_element in
    let result = make repr in
    setf result Fields.type_ type_ ;
    setf result Fields.value (Unsigned.UInt32.of_int value) ;
    result
end

module Timings =
struct
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

  let get (p : Types.Timings.t structure) =
    let open Types.Timings in
    let t_start_ms = getf p Fields.t_start_ms in
    let t_end_ms = getf p Fields.t_end_ms in
    let t_load_ms = getf p Fields.t_load_ms in
    let t_sample_ms = getf p Fields.t_sample_ms in
    let t_p_eval_ms = getf p Fields.t_p_eval_ms in
    let t_eval_ms = getf p Fields.t_eval_ms in
    let n_sample = getf p Fields.n_sample in
    let n_p_eval = getf p Fields.n_p_eval in
    let n_eval = getf p Fields.n_eval in
    {
      t_start_ms  ;
      t_end_ms  ;
      t_load_ms  ;
      t_sample_ms  ;
      t_p_eval_ms  ;
      t_eval_ms  ;
      n_sample  ;
      n_p_eval  ;
      n_eval
    }
end

type model = Types.Model.t Ctypes.structure Ctypes.ptr

type context = Types.Context.t Ctypes.structure Ctypes.ptr

type grammar = Types.Grammar.t Ctypes.structure Ctypes.ptr


let backend_init ~numa = Stubs.backend_init numa

let backend_free = Stubs.backend_free

let load_model_from_file model params =
  let ptr = Stubs.load_model_from_file model params in
  if Ctypes.is_null ptr then
    None
  else
    Some ptr

let free_model = Stubs.free_model

let new_context_with_model = Stubs.new_context_with_model

let free = Stubs.free

let time_us = Stubs.time_us

let max_devices = Stubs.max_devices

let mmap_supported = Stubs.mmap_supported

let mlock_supported = Stubs.mlock_supported

let n_vocab = Stubs.n_vocab

let n_ctx = Stubs.n_ctx

let n_ctx_train = Stubs.n_ctx_train

let n_embd = Stubs.n_embd

let vocab_type = Stubs.vocab_type

let model_n_vocab = Stubs.model_n_vocab

let model_n_ctx = Stubs.model_n_ctx

let model_n_ctx_train = Stubs.model_n_ctx_train

let model_n_embd = Stubs.model_n_embd

let model_desc model =
  let wants_to_write = Stubs.model_desc model Ctypes.(from_voidp char null) Unsigned.Size_t.zero in
  (* +1 for null character *)
  let buff = Array1.create Char c_layout (wants_to_write + 1) in
  let buff_ptr = Ctypes.bigarray_start Ctypes.array1 buff in
  let wrote = Stubs.model_desc model buff_ptr Unsigned.Size_t.zero in
  if wrote <> wants_to_write then
    failwith "model_desc: error while writing model description" ;
  String.init wrote (fun i -> buff.{i})

let model_size model = Stubs.model_size model |> Unsigned.UInt64.to_int

let model_n_params model = Stubs.model_n_params model |> Unsigned.UInt64.to_int

let model_quantize ~fname_inp ~fname_out params =
  let fname_inp = CArray.of_string fname_inp |> CArray.start in
  let fname_out = CArray.of_string fname_out |> CArray.start in
  let retcode = Stubs.model_quantize fname_inp fname_out params in
  retcode = 0

let model_apply_lora_from_file model ~path_lora ~path_base_model ~n_threads =
  let path_lora = CArray.of_string path_lora |> CArray.start in
  let path_base_model = CArray.of_string path_base_model |> CArray.start in
  let retcode = Stubs.model_apply_lora_from_file model path_lora path_base_model n_threads in
  retcode = 0

let get_kv_cache_token_count = Stubs.get_kv_cache_token_count

let set_rng_seed context seed = Stubs.set_rng_seed context (Unsigned.UInt32.of_int seed)

let get_state_size context = Stubs.get_state_size context |> Unsigned.Size_t.to_int

type buff = (char, int8_unsigned_elt, c_layout) Array1.t

let copy_state_data context (buff : buff) =
  let ptr =
    Ctypes.bigarray_start Ctypes.array1 buff
    |> Ctypes.to_voidp
    |> Ctypes.from_voidp Ctypes.uint8_t
  in
  Stubs.copy_state_data context ptr |> Unsigned.Size_t.to_int

let set_state_data context (buff : buff) =
  let ptr =
    Ctypes.bigarray_start Ctypes.array1 buff
    |> Ctypes.to_voidp
    |> Ctypes.from_voidp Ctypes.uint8_t
  in
  Stubs.set_state_data context ptr |> Unsigned.Size_t.to_int

let load_session_file context ~path_session (tokens : Token_buffer.t) =
  let path_session = CArray.of_string path_session |> CArray.start in
  let n_token_count_out = Ctypes.allocate size_t Unsigned.Size_t.zero in
  let token_buff_ptr = Ctypes.bigarray_start Ctypes.array1 tokens in
  let token_buff_len = Array1.dim tokens |> Unsigned.Size_t.of_int in
  if Stubs.load_session_file context path_session token_buff_ptr token_buff_len n_token_count_out then
    Ctypes.(!@ n_token_count_out)
    |> Unsigned.Size_t.to_int
    |> Option.some
  else
    None

let save_session_file context ~path_session (tokens : Token_buffer.t) =
  let path_session = CArray.of_string path_session |> CArray.start in
  let token_buff_ptr = Ctypes.bigarray_start Ctypes.array1 tokens in
  let token_buff_len = Array1.dim tokens |> Unsigned.Size_t.of_int in
  Stubs.save_session_file context path_session token_buff_ptr token_buff_len

let eval context (tokens : token_buff) ~n_tokens ~n_past ~n_threads =
  let token_buff_ptr = Ctypes.bigarray_start Ctypes.array1 tokens in
  let eval_success = Stubs.eval context token_buff_ptr n_tokens n_past n_threads = 0 in
  if eval_success then
    let ptr = Stubs.get_logits context in
    let n_vocab = n_vocab context in
    let ba = Ctypes.bigarray_of_ptr array2 (n_tokens, n_vocab) Float32 ptr in
    Some ba
  else
    None

let eval_embd context (embd : (float, float32_elt, c_layout) Array1.t) ~n_tokens ~n_past ~n_threads =
  let ptr = Ctypes.bigarray_start Ctypes.array1 embd in
  let eval_success = Stubs.eval_embd context ptr n_tokens n_past n_threads = 0 in
  if eval_success then
    let ptr = Stubs.get_logits context in
    let n_vocab = n_vocab context in
    let ba = Ctypes.bigarray_of_ptr Ctypes.array2 (n_tokens, n_vocab) Float32 ptr in
    Some ba
  else
    None

let eval_export context fname =
  let fname = CArray.of_string fname |> CArray.start in
  Stubs.eval_export context fname = 0

let get_embeddings context =
  let ptr = Stubs.get_embeddings context in
  let n_embd = n_embd context in
  Ctypes.bigarray_of_ptr array1 n_embd Float32 ptr

let token_get_text context token =
  let ptr = Stubs.token_get_text context token in
  let length = Stubs.strlen ptr |> Unsigned.Size_t.to_int in
  Ctypes.string_from_ptr ptr ~length

let token_get_score = Stubs.token_get_score

let token_get_type = Stubs.token_get_type

let token_bos = Stubs.token_bos

let token_eos = Stubs.token_eos

let token_nl = Stubs.token_nl

let tokenize context ~text tokens ~n_max_tokens ~add_bos =
  if n_max_tokens <= 0 then
    invalid_arg "tokenize (n_max_tokens <= 0)" ;
  let text = CArray.of_string text |> CArray.start in
  let tokens = Ctypes.bigarray_start Ctypes.array1 tokens in
  let written = Stubs.tokenize context text tokens n_max_tokens add_bos in
  if written < 0 then
    Error (`Too_many_tokens written)
  else
    Ok written

let tokenize_with_model model ~text tokens ~n_max_tokens ~add_bos =
  if n_max_tokens <= 0 then
    invalid_arg "tokenize (n_max_tokens <= 0)" ;
  let text = CArray.of_string text |> CArray.start in
  let tokens = Ctypes.bigarray_start Ctypes.array1 tokens in
  let written = Stubs.tokenize_with_model model text tokens n_max_tokens add_bos in
  if written < 0 then
    Error (`Too_many_tokens written)
  else
    Ok written

let token_to_piece context token =
  let rec loop len =
    let buff = CArray.make char len in
    let ptr = CArray.start buff in
    let written = Stubs.token_to_piece context token ptr len in
    if written < 0 then
      (* Buffer too small *)
      loop (- written)
    else
      Ctypes.string_from_ptr ptr ~length:written
      |> Result.ok
  in
  loop 32

let token_to_piece_with_model model token =
  let rec loop len =
    let buff = CArray.make char len in
    let ptr = CArray.start buff in
    let written = Stubs.token_to_piece_with_model model token ptr len in
    if written = 0 then
      Error `Invalid_token
    else if written < 0 then
      (* Buffer too small *)
      loop (- written)
    else
      Ctypes.string_from_ptr ptr ~length:written
      |> Result.ok
  in
  loop 32

let grammar_init (rules : Grammar_element.t array array) ~start_rule_index =
  let n_rules = Array.length rules in
  let ptr =
    Array.map (fun rule ->
        rule
        |> Array.to_seq
        |> Seq.map Grammar_element.make_internal
        |> List.of_seq
        |> CArray.of_list Types.Grammar_element.repr
        |> CArray.start
      ) rules
    |> Array.to_list
    |> CArray.of_list Ctypes.(ptr Types.Grammar_element.repr)
    |> CArray.start
  in
  let result =
    Stubs.grammar_init ptr (Unsigned.Size_t.of_int n_rules) (Unsigned.Size_t.of_int start_rule_index)
  in
  Gc.finalise Stubs.grammar_free result ;
  result

let grammar_copy grammar =
  let grammar = Stubs.grammar_copy grammar in
  Gc.finalise Stubs.grammar_free grammar ;
  grammar

let sample_repetition_penalty context ~candidates ~last_tokens ~penalty =
  let dim = Array1.dim last_tokens |> Unsigned.Size_t.of_int in
  Stubs.sample_repetition_penalty context candidates (Ctypes.bigarray_start array1 last_tokens) dim penalty

let sample_frequency_and_presence_penalties context ~candidates ~last_tokens ~alpha_frequency ~alpha_presence =
  let dim = Array1.dim last_tokens |> Unsigned.Size_t.of_int in
  Stubs.sample_frequency_and_presence_penalties context candidates (Ctypes.bigarray_start array1 last_tokens) dim alpha_frequency alpha_presence

let sample_classifier_free_guidance context ~candidates ~guidance_ctx ~scale =
  Stubs.sample_classifier_free_guidance context candidates guidance_ctx scale

let sample_softmax context ~candidates =
  Stubs.sample_softmax context candidates

let sample_top_k context ~candidates ~k ~min_keep =
  Stubs.sample_top_k context candidates k (Unsigned.Size_t.of_int min_keep)

let sample_top_p context ~candidates ~p ~min_keep =
  Stubs.sample_top_p context candidates p (Unsigned.Size_t.of_int min_keep)

let sample_tail_free context ~candidates ~z ~min_keep =
  Stubs.sample_tail_free context candidates z (Unsigned.Size_t.of_int min_keep)

let sample_typical context ~candidates ~p ~min_keep =
  Stubs.sample_typical context candidates p (Unsigned.Size_t.of_int min_keep)

let sample_temperature context ~candidates ~temp =
  Stubs.sample_temperature context candidates temp

let sample_grammar context ~candidates grammar =
  Stubs.sample_grammar context candidates grammar

let sample_token_mirostat context ~candidates ~tau ~eta ~m =
  let mu = Ctypes.allocate float 0.0 in
  let token = Stubs.sample_token_mirostat context candidates tau eta m mu in
  (!@mu, token)

let sample_token_mirostat_v2 context ~candidates ~tau ~eta =
  let mu = Ctypes.allocate float 0.0 in
  let token = Stubs.sample_token_mirostat_v2 context candidates tau eta mu in
  (!@mu, token)

let sample_token_greedy context ~candidates =
  Stubs.sample_token_greedy context candidates

let sample_token context ~candidates =
  Stubs.sample_token_greedy context candidates

let grammar_accept_token = Stubs.grammar_accept_token

type beam_view = {
  tokens : token_buff ;
  p : float ;
  eob : bool
}

type beam_search_callback =
  beam_views:beam_view array ->
  common_prefix_length:int ->
  last_call:bool ->
  unit

let beam_search context callback ~n_beams ~n_past ~n_predict ~n_threads =
  let callback : (unit Ctypes_static.ptr -> Types.Beams_state.t structure -> unit) =
    fun _null beams_state ->
      let open Types.Beams_state in
      let beam_views = getf beams_state Fields.beam_views in
      let n_beams =
        getf beams_state Fields.n_beams
        |> Unsigned.Size_t.to_int in
      let common_prefix_length =
        getf beams_state Fields.common_prefix_length
        |> Unsigned.Size_t.to_int in
      let last_call = getf beams_state Fields.last_call in
      let beam_views =
        CArray.from_ptr beam_views n_beams
        |> CArray.to_list
        |> List.map (fun view ->
            let open Types.Beam_view in
            let tokens = getf view Fields.tokens in
            let n_token =
              getf view Fields.n_token
              |> Unsigned.Size_t.to_int in
            let tokens = Ctypes.bigarray_of_ptr array1 n_token Int32 tokens in
            let p = getf view Fields.p in
            let eob = getf view Fields.eob in
            { tokens; p; eob }
          )
        |> Array.of_list
      in
      callback
        ~beam_views
        ~common_prefix_length
        ~last_call
  in
  Stubs.beam_search context callback Ctypes.null (Unsigned.Size_t.of_int n_beams) n_past n_predict n_threads

let get_timings context =
  let timings = Stubs.get_timings context in
  Timings.get timings

let print_timings = Stubs.print_timings

let reset_timings = Stubs.reset_timings

let print_system_info () =
  let ptr = Stubs.print_system_info () in
  let length = Stubs.strlen ptr |> Unsigned.Size_t.to_int in
  Ctypes.string_from_ptr ptr ~length

let log_set callback =
  let callback : Log_level.t -> char Ctypes_static.ptr -> unit Ctypes_static.ptr -> unit =
    fun log_level ptr _null ->
      let length = Stubs.strlen ptr |> Unsigned.Size_t.to_int in
      let string = Ctypes.string_from_ptr ptr ~length in
      callback log_level string
  in
  Stubs.log_set callback Ctypes.null
