open Bigarray

let () = Llama_cpp.log_set (fun _ _ -> ())

let main model prompt =
  let n_len = 32 in
  Llama_cpp.backend_init ~numa:false ;
  (* Initialize the model*)
  let model_params = Llama_cpp.Model_params.default () in
  let model =
    match Llama_cpp.load_model_from_file model model_params with
    | None ->
      Printf.eprintf "%s: error: unable to load model\n" __FUNCTION__ ;
      exit 1
    | Some model -> model
  in
  let ctx_params = Llama_cpp.Context_params.default () in
  Llama_cpp.Context_params.(
    (* TODO: query actual number of physical cores *)
    let n_threads = 8 in
    set_seed ctx_params 1234 ;
    set_n_ctx ctx_params 2048 ;
    set_n_threads ctx_params n_threads ;
    set_n_threads_batch ctx_params n_threads
  ) ;
  let ctx = Llama_cpp.new_context_with_model model ctx_params in
  (* Tokenize the prompt *)
  let n_max_tokens =
    (* upper limit for the number of tokens, + 1 for bos *)
    String.length prompt + 1
  in
  let tokens_list = Array1.create Int32 c_layout n_max_tokens in
  let tokens_list =
    match Llama_cpp.tokenize model ~text:prompt tokens_list ~n_max_tokens ~add_bos:true with
    | Error (`Too_many_tokens count) ->
      Printf.eprintf "%s: error: too many tokens (got %d, expected <= %d)" __FUNCTION__ count n_max_tokens ;
      exit 1
    | Ok written ->
      Array1.sub tokens_list 0 written
  in
  let n_ctx = Llama_cpp.n_ctx ctx in
  let n_kv_req = Array1.dim tokens_list + (n_len - Array1.dim tokens_list) in
  (* Make sure KV cache is big enough to hold all the prompt and generated tokens *)
  if n_kv_req > n_ctx then (
    Printf.eprintf "%s: error: n_kv_req > n_ctx, the required KV cache size is not big enough\n" __FUNCTION__ ;
    Printf.eprintf "%s:        either reduce n_parallel or increase n_ctx\n" __FUNCTION__)
  else () ;
  (* Print the prompt token-by-token *)
  Printf.eprintf "\n" ;
  for i = 0 to Array1.dim tokens_list - 1 do
    let token = tokens_list.{i} in
    match Llama_cpp.token_to_piece model token with
    | Ok s ->
      Printf.printf "%s" s
    | Error `Invalid_token ->
      Printf.eprintf "%ld: invalid token\n" token
  done ;
  Printf.eprintf "%!" ;

  (* Create a llama_batch with size 512 *)
  (* we use this object to submit token data for decoding *)
  Llama_cpp.with_batch ~n_tokens:512 ~embd:0 @@ fun batch ->
  Llama_cpp.Batch.set_n_tokens batch (Array1.dim tokens_list) ;
  let Llama_cpp.Batch.{ n_tokens; token; pos; seq_id; logits = compute_logits; embd = _ } = Llama_cpp.Batch.view batch in

  (* evaluate the initial prompt *)
  for i = 0 to n_tokens - 1 do
    token.{i} <- tokens_list.{i} ;
    pos.{i} <- Int32.of_int i ;
    seq_id.{i} <- 0l ;
    compute_logits.{i} <- 0 (* false *)
  done ;

  (* llama_decode will output logits only for the last token of the prompt *)
  compute_logits.{n_tokens - 1} <- 1 ; (* true *)

  let logits =
    match Llama_cpp.decode ctx batch with
    | Ok logits -> logits
    | Error `no_kv_slot_for_batch | Error `decode_error ->
      Printf.eprintf "%s: Llama_cpp.decode failed" __FUNCTION__ ;
      exit 1
  in

  (* main loop *)

  (* The LLM keeps a contextual cache memory of previous token evaluation. *)
  (* Usually, once this cache is full, it is required to recompute a compressed context based on previous *)
  (* tokens (see "infinite text generation via context swapping" in the main example), but in this minimalist *)
  (* example, we will just stop the loop once this cache is full or once an end of stream is detected. *)

  let rec loop prev_logits n_cur n_decode =
    if n_cur > n_len then ()
    else
      (
        let logits = Array2.slice_left prev_logits (Llama_cpp.Batch.n_tokens batch - 1) in
        (* sample the next token *)
        let candidates = Llama_cpp.Token_data_array.create logits in
        let new_token_id = Llama_cpp.sample_token_greedy ctx ~candidates in
        (* is it an end of stream ? *)
        if new_token_id = Llama_cpp.token_eos ctx then
          Printf.eprintf "[end of text]\n%!"
        else
          (
            (match Llama_cpp.token_to_piece model new_token_id with
             | Ok s ->
               Printf.printf "%s%!" s ;
             | Error `Invalid_token ->
               Printf.eprintf "%s : invalid token (%ld)\n" __FUNCTION__ new_token_id ;
               exit 1
            ) ;
            (* prepare the next batch *)
            Llama_cpp.Batch.set_n_tokens batch 0 ;
            token.{0} <- new_token_id ;
            pos.{0} <- Int32.of_int n_cur ;
            seq_id.{0} <- 0l ;
            compute_logits.{0} <- 1 ; (* true *)
            Llama_cpp.Batch.set_n_tokens batch 1 ;
            let n_decode = n_decode + 1 in
            let n_cur = n_cur + 1 in
            let logits =
              match Llama_cpp.decode ctx batch with
              | Ok logits -> logits
              | Error _ ->
                Printf.eprintf "%s : failed to eval\n" __FUNCTION__ ;
                exit 1
            in
            loop logits n_cur n_decode
          )
      )
  in loop logits (Llama_cpp.Batch.n_tokens batch) 0

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
