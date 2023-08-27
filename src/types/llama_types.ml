module Make = functor (T : Cstubs_structs.TYPE) ->
struct
  open Ctypes_static
  open T

  module Progress_callback =
  struct
    type t

    let repr : (float -> unit Ctypes_static.ptr -> unit) static_funptr typ
      = static_funptr (float @-> (ptr void) @-> returning void)
  end

  module Context_params =
  struct
    type t

    let repr : t structure typ = structure "llama_context_params"

    module Fields =
    struct
      let seed  = field repr "seed" uint32_t
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

    let repr : t structure typ = structure "llama_context"
  end

  module Model =
  struct
    type t

    let repr :  t structure typ = structure "llama_model"
  end
end
