open Bigarray

(* TODO: query actual number of physical cores *)
let n_threads = 8

let main model prompt =
  Llama_cpp.backend_init ~numa:false ;
  let ctx_params = Llama_cpp.Context_params.default () in
  let model =
    match Llama_cpp.load_model_from_file model ctx_params with
    | None ->
      Printf.eprintf "%s: error: unable to load model\n" __FUNCTION__ ;
      exit 1
    | Some model -> model
  in
  let ctx = Llama_cpp.new_context_with_model model ctx_params in
  (* Tokenize the prompt *)
  let n_max_tokens = 1024 in
  let tokens_list = Array1.create Int32 c_layout n_max_tokens in
  let tokens_list =
    match Llama_cpp.tokenize ctx ~text:prompt tokens_list ~n_max_tokens ~add_bos:true with
    | Error (`Too_many_tokens count) ->
      Printf.eprintf "%s: error: too many tokens (got %d, expected <= %d)" __FUNCTION__ count n_max_tokens ;
      exit 1
    | Ok written ->
      Array1.sub tokens_list 0 written
  in
  let max_context_size = Llama_cpp.n_ctx ctx in
  let max_tokens_list_size = max_context_size - 4 in
  if Array1.dim tokens_list > max_tokens_list_size then
    Printf.eprintf
      "%s: error: prompt too long (%d tokens, max %d)\n"
      __FUNCTION__
      (Array1.dim tokens_list)
      max_tokens_list_size
  else () ;
  Printf.eprintf "\n\n" ;
  for i = 0 to Array1.dim tokens_list - 1 do
    let token = tokens_list.{i} in
    match Llama_cpp.token_to_piece ctx token with
    | Ok s ->
      Printf.eprintf "%s" s
    | Error `Invalid_token ->
      Printf.eprintf "%ld: invalid token\n" token
  done ;
  Printf.eprintf "%!" ;

  (* main loop *)

  (* The LLM keeps a contextual cache memory of previous token evaluation. *)
  (* Usually, once this cache is full, it is required to recompute a compressed context based on previous *)
  (* tokens (see "infinite text generation via context swapping" in the main example), but in this minimalist *)
  (* example, we will just stop the loop once this cache is full or once an end of stream is detected. *)

  let n_gen = Int.min 32 max_context_size in

  let rec loop ~n_tokens =
    if Llama_cpp.get_kv_cache_token_count ctx >= n_gen then
      Printf.eprintf "[out of context]\n%!"
    else
      (
        (* Evaluate the transformer *)
        let logits =
          match
            Llama_cpp.eval
              ctx
              tokens_list
              ~n_tokens
              ~n_past:(Llama_cpp.get_kv_cache_token_count ctx)
              ~n_threads
          with
          | None ->
            Printf.eprintf "%s : failed to eval\n" __FUNCTION__ ;
            exit 1
          | Some logits -> logits
        in
        let logits = Array2.slice_left logits 0 in
        (* sample the next token *)
        let candidates = Llama_cpp.Token_data_array.create logits in
        let new_token_id = Llama_cpp.sample_token_greedy ctx ~candidates in
        (* is it an end of stream ? *)
        if new_token_id = Llama_cpp.token_eos ctx then
          Printf.eprintf "[end of text]\n%!"
        else
          (* print the new token *)
          (
            (match Llama_cpp.token_to_piece ctx new_token_id with
             | Ok s ->
               Printf.printf "%s%!" s ;
             | Error `Invalid_token ->
               Printf.eprintf "%s : invalid token (%ld)\n" __FUNCTION__ new_token_id ;
               exit 1
            ) ;
            tokens_list.{0} <- new_token_id ;
            loop ~n_tokens:1
          )
      )
  in loop ~n_tokens:(Array1.dim tokens_list)



let usage () =
  Printf.printf "usage: %s MODEL_PATH [PROMPT]\n" Sys.argv.(0) ;
  exit 1

let default_prompt = "Hello my name is"

let () =
  match Array.to_list Sys.argv |> List.tl with
  | [model] ->
    main model default_prompt
  | [model; prompt] ->
    main model prompt
  | _ -> usage ()
