open Llama_fixed_types.Types

module Make = functor (T : Cstubs.FOREIGN) -> struct
  open Ctypes
  open T

  let load_model_from_file =
    foreign "llama_load_model_from_file" (string @-> Context_params.repr @-> returning (ptr Model.repr))

  let free_model =
    foreign "llama_free_model" (ptr Model.repr @-> returning void)

  let new_context_with_model =
    foreign "llama_new_context_with_model" (ptr Model.repr @-> Context_params.repr @-> returning (ptr Context.repr))

  let free =
    foreign "llama_free" (ptr Context.repr @-> returning void)
end
