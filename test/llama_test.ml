open Llama_cpp
open Bigarray

(* Testing token data arrays *)
module Token_data_array_test =
struct
  open Token_data_array

  let logits = Array1.of_array Float32 c_layout [| 0.0; 1.0; 2.0 |]

  let arr = create logits
end

let _cfg =
  Llama_cpp.Context_params.make
    ~seed:42
    ~n_ctx:512
    ~n_batch:1
    ~n_threads:1
    ~n_threads_batch:1
    ~rope_freq_base:1.
    ~rope_freq_scale:1.
    ~mul_mat_q:false
    ~f16_kv:false
    ~logits_all:false
    ~embedding:false
