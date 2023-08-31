module Make = functor (T : Ctypes.TYPE) ->
struct
  open T

  module Log_level =
  struct
    type t =
      | Error
      | Warn
      | Info

    let vals =
      [
        (Error, constant "LLAMA_LOG_LEVEL_ERROR" int64_t);
        (Warn, constant "LLAMA_LOG_LEVEL_WARN" int64_t);
        (Info, constant "LLAMA_LOG_LEVEL_INFO" int64_t);
      ]

    let repr = enum ~typedef:false "llama_log_level" vals
  end

  module Progress_callback =
  struct
    type t

    let repr : (float -> unit Ctypes.ptr -> unit) static_funptr typ
      = static_funptr (float @-> (ptr void) @-> returning void)
  end

  module Token =
  struct
    type t

    let repr = typedef int "llama_token"
  end

  module Context_params =
  struct
    type t

    let repr : t Ctypes.structure typ = structure "llama_context_params"

    module Fields =
    struct
      let seed = field repr "seed" uint32_t
      let n_ctx = field repr "n_ctx" int32_t
      let n_batch = field repr "n_batch" int32_t
      let n_gpu_layers = field repr "n_gpu_layers" int32_t
      let main_gpu = field repr "main_gpu" int32_t

      let tensor_split = field repr "tensor_split" (ptr float)

      let rope_freq_base = field repr "rope_freq_base" float
      let rope_freq_scale = field repr "rope_freq_scale" float

      let progress_callback = field repr "progress_callback" Progress_callback.repr
      let progress_callback_user_data = field repr "progress_callback_user_data" (ptr void)

      let low_vram = field repr "low_vram" bool
      let mul_mat_q = field repr "mul_mat_q" bool
      let f16_kv = field repr "f16_kv" bool
      let logits_all = field repr "logits_all" bool
      let vocab_only = field repr "vocab_only" bool
      let use_mmap = field repr "use_mmap" bool
      let use_mlock = field repr "use_mlock" bool
      let embedding = field repr "embedding" bool
      let () = seal repr
    end
  end

  module Context =
  struct
    type t

    let repr : t Ctypes.structure typ = structure "llama_context"
  end

  module Model =
  struct
    type t

    let repr :  t Ctypes.structure typ = structure "llama_model"
  end

  module Vocab_type =
  struct
    type t =
      | Spm (** SentencePiece *)
      | Bpe (** Byte Pair Encoding *)

    let vals =
      [
        (Spm, constant "LLAMA_VOCAB_TYPE_SPM" int64_t);
        (Bpe, constant "LLAMA_VOCAB_TYPE_BPE" int64_t)
      ]

    let repr = enum ~typedef:false "llama_vocab_type" vals
  end

  module Token_type =
  struct
    type t =
      | Undefined
      | Normal
      | Unknown
      | Control
      | User_defined
      | Unused
      | Byte

    let vals =
      [
        (Undefined, constant "LLAMA_TOKEN_TYPE_UNDEFINED" int64_t);
        (Normal, constant "LLAMA_TOKEN_TYPE_NORMAL" int64_t);
        (Unknown, constant "LLAMA_TOKEN_TYPE_UNKNOWN" int64_t);
        (Control, constant "LLAMA_TOKEN_TYPE_CONTROL" int64_t);
        (User_defined, constant "LLAMA_TOKEN_TYPE_USER_DEFINED" int64_t);
        (Unused, constant "LLAMA_TOKEN_TYPE_UNUSED" int64_t);
        (Byte, constant "LLAMA_TOKEN_TYPE_BYTE" int64_t);
      ]

    let repr = enum ~typedef:false "llama_token_type" vals
  end

  module File_type =
  struct
    type t =
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

    let vals =
      [
        (ALL_F32, constant "LLAMA_FTYPE_ALL_F32" int64_t);
        (MOSTLY_F16, constant "LLAMA_FTYPE_MOSTLY_F16" int64_t);
        (MOSTLY_Q4_0, constant "LLAMA_FTYPE_MOSTLY_Q4_0" int64_t);
        (MOSTLY_Q4_1, constant "LLAMA_FTYPE_MOSTLY_Q4_1" int64_t);
        (MOSTLY_Q4_1_SOME_F16, constant "LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16" int64_t);
        (MOSTLY_Q8_0, constant "LLAMA_FTYPE_MOSTLY_Q8_0" int64_t);
        (MOSTLY_Q5_0, constant "LLAMA_FTYPE_MOSTLY_Q5_0" int64_t);
        (MOSTLY_Q5_1, constant "LLAMA_FTYPE_MOSTLY_Q5_1" int64_t);
        (MOSTLY_Q2_K, constant "LLAMA_FTYPE_MOSTLY_Q2_K" int64_t);
        (MOSTLY_Q3_K_S, constant "LLAMA_FTYPE_MOSTLY_Q3_K_S" int64_t);
        (MOSTLY_Q3_K_M, constant "LLAMA_FTYPE_MOSTLY_Q3_K_M" int64_t);
        (MOSTLY_Q3_K_L, constant "LLAMA_FTYPE_MOSTLY_Q3_K_L" int64_t);
        (MOSTLY_Q4_K_S, constant "LLAMA_FTYPE_MOSTLY_Q4_K_S" int64_t);
        (MOSTLY_Q4_K_M, constant "LLAMA_FTYPE_MOSTLY_Q4_K_M" int64_t);
        (MOSTLY_Q5_K_S, constant "LLAMA_FTYPE_MOSTLY_Q5_K_S" int64_t);
        (MOSTLY_Q5_K_M, constant "LLAMA_FTYPE_MOSTLY_Q5_K_M" int64_t);
        (MOSTLY_Q6_K, constant "LLAMA_FTYPE_MOSTLY_Q6_K" int64_t);
        (GUESSED, constant "LLAMA_FTYPE_GUESSED" int64_t);
      ]

    let repr = enum ~typedef:false "llama_ftype" vals
  end

  module Token_data =
  struct
    type t

    let repr : t Ctypes.structure typ = structure "llama_token_data"

    module Fields =
    struct
      let id = field repr "id" Token.repr

      let logit = field repr "logit" float

      let p = field repr "p" float

      let () = seal repr
    end
  end

  module Token_data_array =
  struct
    type t

    let repr : t Ctypes.structure typ = structure "llama_token_data_array"

    module Fields =
    struct
      let data = field repr "data" (ptr Token_data.repr)

      let size = field repr "size" size_t

      let sorted = field repr "sorted" bool

      let () = seal repr
    end
  end

  module Model_quantize_params =
  struct
    type t

    let repr : t Ctypes.structure typ = structure "llama_model_quantize_params"

    module Fields =
    struct
      let nthread = field repr "nthread" int

      let ftype = field repr "ftype" File_type.repr

      let allow_requantize = field repr "allow_requantize" bool

      let quantize_output_tensor = field repr "quantize_output_tensor" bool

      let () = seal repr
    end
  end

  module Grammar =
  struct
    type t

    let repr : t Ctypes.structure typ = structure "llama_grammar"
  end

  module Gretype =
  struct
    type t =
        (* end of rule definition *)
        | GRETYPE_END

        (* start of alternate definition for rule *)
        | GRETYPE_ALT

        (* non-terminal element: reference to rule *)
        | GRETYPE_RULE_REF

        (* terminal element: character (code point) *)
        | GRETYPE_CHAR

        (* inverse char(s) ([^a], [^a-b] [^abc]) *)
        | GRETYPE_CHAR_NOT

        (* modifies a preceding | GRETYPE_CHAR or LLAMA_GRETYPE_CHAR_ALT to *)
        (* be an inclusive range ([a-z]) *)
        | GRETYPE_CHAR_RNG_UPPER

        (* modifies a preceding | GRETYPE_CHAR or *)
        (* LLAMA_GRETYPE_CHAR_RNG_UPPER to add an alternate char to match ([ab], [a-zA]) *)
        | GRETYPE_CHAR_ALT


    let vals =
      [
        (GRETYPE_END, constant "LLAMA_GRETYPE_END" int64_t);
        (GRETYPE_ALT, constant "LLAMA_GRETYPE_ALT" int64_t);
        (GRETYPE_RULE_REF, constant "LLAMA_GRETYPE_RULE_REF" int64_t);
        (GRETYPE_CHAR, constant "LLAMA_GRETYPE_CHAR" int64_t);
        (GRETYPE_CHAR_NOT, constant "LLAMA_GRETYPE_CHAR_NOT" int64_t);
        (GRETYPE_CHAR_RNG_UPPER, constant "LLAMA_GRETYPE_CHAR_RNG_UPPER" int64_t);
        (GRETYPE_CHAR_ALT, constant "LLAMA_GRETYPE_CHAR_ALT" int64_t)
      ]

    let repr = enum ~typedef:false "llama_gretype" vals
  end

  module Grammar_element =
  struct
    type t

    let repr : t Ctypes.structure typ =
      typedef (structure "llama_grammar_element") "llama_grammar_element"

    module Fields =
    struct
      let type_ = field repr "type" Gretype.repr

      let value = field repr "value" uint32_t

      let () = seal repr
    end
  end

  module Timings =
  struct
    type t

    let repr : t Ctypes.structure typ = structure "llama_timings"

    module Fields =
    struct
      let t_start_ms = field repr "t_start_ms" double
      let t_end_ms = field repr "t_end_ms" double
      let t_load_ms = field repr "t_load_ms" double
      let t_sample_ms = field repr "t_sample_ms" double
      let t_p_eval_ms = field repr "t_p_eval_ms" double
      let t_eval_ms = field repr "t_eval_ms" double

      let n_sample = field repr "n_sample" int32_t
      let n_p_eval = field repr "n_p_eval" int32_t
      let n_eval = field repr "n_eval" int32_t

      let () = seal repr
    end
  end

  module Beam_view =
  struct
    type t

    let repr : t Ctypes.structure typ = structure "llama_beam_view"

    module Fields =
    struct
      let tokens = field repr "tokens" (ptr Token.repr)

      let n_token = field repr "n_tokens" size_t

      (* Cumulative beam probability (renormalized relative to all beams) *)
      let p = field repr "p" float

      (* Callback should set this to true when a beam is at end-of-beam. *)
      let eob = field repr "eob" bool

      let () = seal repr
    end
  end

  module Beams_state =
  struct
    (* Passed to beam_search_callback function. *)
    (* Whenever 0 < common_prefix_length, this number of tokens should be copied from any of the beams *)
    (* (e.g. beams[0]) as they will be removed (shifted) from all beams in all subsequent callbacks. *)
    (* These pointers are valid only during the synchronous callback, so should not be saved. *)
    type t

    let repr : t Ctypes.structure typ = structure "llama_beams_state"

    module Fields =
    struct
      let beam_views = field repr "beam_views" (ptr Beam_view.repr)

      let n_beams = field repr "n_beams" size_t

      let common_prefix_length = field repr "common_prefix_length" size_t

      let last_call = field repr "last_call" bool

      let () = seal repr
    end
  end
end
