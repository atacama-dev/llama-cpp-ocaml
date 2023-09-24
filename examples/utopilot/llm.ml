open Bigarray

module Token_buffer = Llama_cpp.Token_buffer

module Hash_set : sig
  type t

  val create : int -> t
  val add : t -> string -> unit
  val mem : t -> string -> bool
  val card : t -> int
end =
  struct
    module T = Hashtbl.Make (String)

    type t = unit T.t
    let create = T.create
    let add table s = T.add table s ()
    let mem = T.mem
    let card = T.length
  end

type token = Llama_cpp.token

type state =
  { n : int ; (* Index of repl interaction *)
    ctx : Llama_cpp.context;
    n_keep : int; (* Number of tokens to keep when resetting context *)
    n_batch : int; (* Max size of a batch *)
    n_threads : int; (* Number of parallel threads to use to perform inference. *)
    n_max_samples : int; (* Maximum sampling budget per user input *)
    n_past : int;
    embd : token list;
    last_tokens : token list;
    last_logits : Llama_cpp.logits option;
    processed_hist : Hash_set.t
  }

let make_state ?(n_keep = 32) ?(n_batch = 32) ?(n_threads = 8) ?(n_max_samples = 512) ctx =
  {
    n = 1 ;
    ctx ;
    n_keep ;
    n_batch ;
    n_threads ;
    n_max_samples ;
    n_past = 0 ;
    embd = [] ;
    last_tokens = [] ;
    last_logits = None ;
    processed_hist = Hash_set.create 41
  }

let clone state model parameters =
  { state with
    ctx = Llama_cpp.clone state.ctx model parameters }

(* Tokenize prompt *)
let tokenize ~add_bos ctx text =
  let tokens_buff = Token_buffer.init 1024 (Fun.const Llama_cpp.zero_token) in
  match Llama_cpp.tokenize ctx ~text tokens_buff ~n_max_tokens:1024 ~add_bos with
  | Error (`Too_many_tokens count) ->
    let tokens_buff = Token_buffer.init count (Fun.const Llama_cpp.zero_token) in
    (match Llama_cpp.tokenize ctx ~text tokens_buff ~n_max_tokens:count ~add_bos:true with
     | Error _ -> failwith "tokenize"
     | Ok written ->
       assert (written = count) ;
       tokens_buff
    )
  | Ok written ->
    Token_buffer.sub tokens_buff 0 written

let context_swapping n_keep n_past embd last_tokens =
  let n_left = n_past - n_keep in
  assert (n_left >= 0) ;
  (* always keep the first token in the context, BOS *)
  let n_past = Int.max 1 n_keep in
  (* copy [n_left/2] past tokens between [n_past] and start of [embd]
     at the start of [embd]. *)
  let embd, _ =
    Iter.(0 -- (n_left / 2 - 1))
    |> Iter.fold (fun (embd, last_tokens) _i ->
        match last_tokens with
        | [] -> (Llama_cpp.zero_token :: embd, last_tokens)
        | token :: last_tokens ->
          (token :: embd, last_tokens)
      ) (embd, last_tokens)
  in
  embd, n_past

let rec sample_batch ctx tokens n_batch n_threads n_past i =
  let len = Token_buffer.dim tokens in
  assert (len > 0) ;
  let batch_size = Int.min len n_batch in
  let res =
    Llama_cpp.eval
      ctx
      tokens
      ~n_tokens:batch_size
      ~n_past
      ~n_threads
  in
  match res with
  | None ->
    failwith "sample_batch"
  | Some logits ->
    let remaining = len - batch_size in
    let n_past = n_past + batch_size in
    if remaining <= 0 then
      (logits, n_past)
    else
      let tokens = Token_buffer.sub tokens batch_size remaining in
      sample_batch ctx tokens n_batch n_threads n_past (i + batch_size)

let token_to_string (state : state) token =
  match Llama_cpp.token_to_piece state.ctx token with
  | Ok s -> s
  | Error `Invalid_token -> failwith "Invalid token"

let tokens_to_string state tokens =
  let buff = Buffer.create 1024 in
  List.iter (fun token ->
      let s = token_to_string state token in
      Buffer.add_string buff s
    ) tokens ;
  Buffer.contents buff

let perform_inference state =
  let n_ctx = Llama_cpp.n_ctx state.ctx in
  let embd, embd_len =
    let embd =
      List.to_seq state.embd
      |> Seq.take n_ctx
      |> List.of_seq
    in
    (embd, List.length embd)
  in
  let embd, n_past =
    if state.n_past + embd_len > n_ctx then
      context_swapping state.n_keep state.n_past embd state.last_tokens
    else embd, state.n_past
  in
  let tokens = Token_buffer.of_list embd in
  let logits, n_past = sample_batch state.ctx tokens state.n_batch state.n_threads n_past 0 in
  { state with
    embd = [] ;
    n_past ;
    last_logits = Some logits
  }

let tokenize_then_inference ~add_bos state input =
  assert (state.embd = []) ;
  let ctx = state.ctx in
  let tokens = tokenize ~add_bos ctx input |> Token_buffer.to_seq |> List.of_seq in
  let state = { state with embd = tokens } in
  if tokens = [] then
    state
  else
    perform_inference state

let perform_inference_on_history state =
  let hist = UTop.stashable_session_history in
  let contents = UTop_history.contents hist |> List.rev in
  List.fold_left (fun state entry ->
      match entry with
      | UTop_history.Input s ->
        (* Add "beginning-of-sentence" token only if we're just starting
           inference, i.e. if processed_hist is empty. *)
        let add_bos = Hash_set.card state.processed_hist = 0 in
        if Hash_set.mem state.processed_hist s then
          state
        else
          (Hash_set.add state.processed_hist s ;
           tokenize_then_inference ~add_bos state s)
      | _ ->
        (* Skip bad inputs and outputs. *)
        state
    ) state contents

(* This produces a transient sequence because of the shared [candidate] array *)
let samples ~add_bos state prompt =
  let ctx = state.ctx in
  let tokens = tokenize ~add_bos ctx prompt |> Token_buffer.to_seq |> List.of_seq in
  let state = { state with embd = tokens } in
  let state =
    if tokens = [] then
      state
    else
      perform_inference state
  in
  let logits = Option.get state.last_logits in
  let logits = Array2.slice_left logits 0 in
  let candidates = Llama_cpp.Token_data_array.create logits in
  let rec loop state () =
    let last_tokens = Token_buffer.of_list (List.rev state.last_tokens) in
    Llama_cpp.sample_repetition_penalty ctx ~candidates ~last_tokens ~penalty:1.1 ;
    let new_token_id = Llama_cpp.sample_token_greedy ctx ~candidates in
    let embd = new_token_id :: state.embd in
    let last_tokens = new_token_id :: state.last_tokens in
    let state = { state with last_tokens; embd } in
    if Int32.equal new_token_id (Llama_cpp.token_eos state.ctx) then
      let next_element = (new_token_id, state) in
      Seq.Cons (next_element, Seq.empty)
    else
      let state = perform_inference state in
      let next_element = (new_token_id, state) in
      let logits = Option.get state.last_logits in
      let logits = Array2.slice_left logits 0 in
      Llama_cpp.Token_data_array.write_logits candidates logits ;
      Seq.Cons (next_element, loop state)
  in
  loop state
