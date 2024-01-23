open Bigarray

module Token_buffer = Llama_cpp.Token_buffer

module Int32_infix =
struct
  let (+) = Int32.add
  let (-) = Int32.sub
  let ( * ) = Int32.mul
  let (/) = Int32.div
end

module Hash_set : sig
  type t

  val create : int -> t
  val add : t -> string -> unit
  val mem : t -> string -> bool
  val card : t -> int
end =
  struct
    module T = Hashtbl.Make (struct include String let hash = Hashtbl.hash end)

    type t = unit T.t
    let create = T.create
    let add table s = T.add table s ()
    let mem = T.mem
    let card = T.length
  end

type token = Llama_cpp.token

type consts = {
  n_keep : int; (* Number of tokens to keep when resetting context *)
  n_batch : int; (* Max size of a batch *)
  n_threads : int; (* Number of parallel threads to use to perform inference. *)
  n_max_samples : int; (* Maximum sampling budget per user input *)
}

type state =
  { 
    consts : consts;
    n : int ; (* Index of repl interaction *)
    ctx : Llama_cpp.context;
    n_past : int; (* Number of tokens in context *)
    last_tokens : token Queue.t; (* Window of tokens on which we check repetition. *)
    last_logits : Llama_cpp.logits option;
    processed_hist : Hash_set.t
  }

let make_state ?(n_keep = 32) ?(n_batch = 32) ?(n_threads = 8) ?(n_max_samples = 512) ctx =
  {
    consts = {
      n_keep ;
      n_batch ;
      n_threads ;
      n_max_samples ;
    } ;
    n = 1 ;
    ctx ;
    n_past = 0 ;
    last_tokens = Queue.create () ;
    last_logits = None ;
    processed_hist = Hash_set.create 41
  }

(* Tokenize prompt *)
let tokenize ~add_bos ctx text : Token_buffer.t =
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

let context_swapping ctx ~n_keep ~n_past =
  let open Int32_infix in
  let n_past = Int32.of_int n_past in
  let n_keep = Int32.of_int n_keep in
  let n_discard = (n_past - n_keep) / 2l in
  Llama_cpp.kv_cache_seq_rm ctx 0l
    ~p0:(n_keep + 1l)
    ~p1:(n_keep + 1l + n_discard) ;
  Llama_cpp.kv_cache_seq_shift ctx 0l
    ~p0:(n_keep + 1l + n_discard)
    ~p1:n_past
    ~delta:(Int32.neg n_discard) ;
  Int32.to_int (n_past - n_discard)

(* [sample_batch ctx n_batch n_past tokens batch] fills [tokens] with new samples.
   Samples are taken by batches. *)
let sample_batch ctx n_batch n_past tokens batch =
  let rec loop start_idx remaining n_past =
    assert (remaining > 0) ;
    let batch_size = Int.min remaining n_batch in
    (* Initialize batch *)
    Llama_cpp.Batch.(
      set_n_tokens batch batch_size ;
      let { token; pos; seq_id; logits; _ } = view batch in
      for i = 0 to batch_size - 1 do
        token.{i} <- tokens.{start_idx + i} ;
        pos.{i} <- Int32.of_int (n_past + i) ;
        seq_id.{i} <- 0l ;
        logits.{i} <- 0
      done ;
      logits.{batch_size - 1} <- 1
    ) ;
    let res = Llama_cpp.decode ctx batch in
    match res with
    | Error _ ->
      failwith "sample_batch"
    | Ok logits ->
      let start_idx = start_idx + batch_size in
      let n_past = n_past + batch_size in
      let remaining = Token_buffer.dim tokens - start_idx in
      if remaining <= 0 then
        (logits, n_past)
      else
        loop start_idx remaining n_past
  in
  loop 0 (Token_buffer.dim tokens) n_past

let token_to_string (state : state) token =
  match Llama_cpp.token_to_piece (Llama_cpp.get_model state.ctx) token with
  | Ok s -> s
  | Error `Invalid_token -> failwith "Invalid token"

let tokens_to_string state tokens =
  let buff = Buffer.create 1024 in
  List.iter (fun token ->
      let s = token_to_string state token in
      Buffer.add_string buff s
    ) tokens ;
  Buffer.contents buff

let perform_inference state embd batch =
  let n_ctx = Llama_cpp.n_ctx state.ctx in
  let embd_len = Token_buffer.dim embd in
  let n_past =
    if state.n_past + embd_len > n_ctx then
      context_swapping state.ctx ~n_keep:state.consts.n_keep ~n_past:state.n_past
    else
      state.n_past
  in
  let logits, n_past = sample_batch state.ctx state.consts.n_batch n_past embd batch in
  { state with
    n_past ;
    last_logits = Some logits
  }

let tokenize_then_inference ~add_bos state batch input =
  let ctx = state.ctx in
  let tokens = tokenize ~add_bos (Llama_cpp.get_model ctx) input in
  if Array1.dim tokens = 0 then
    state
  else
    perform_inference state tokens batch

let perform_inference_on_history state =
  let batch = Llama_cpp.batch_init ~n_tokens:state.consts.n_batch ~embd:0 in
  let hist = UTop.stashable_session_history in
  let contents = UTop_history.contents hist |> List.rev in
  let state =
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
             tokenize_then_inference ~add_bos state batch s)
        | _ ->
          (* Skip bad inputs and outputs. *)
          state
      ) state contents
  in
  Llama_cpp.batch_free batch ;
  state

(* This produces a transient sequence because of the shared [candidate] array *)
let samples ~add_bos state prompt =
  let ctx = state.ctx in
  let tokens = tokenize ~add_bos (Llama_cpp.get_model state.ctx) prompt in
  let batch = Llama_cpp.batch_init ~n_tokens:state.consts.n_batch ~embd:0 in
  let state =
    if Token_buffer.dim tokens = 0 then
      state
    else
      perform_inference state tokens batch
  in
  let logits = Option.get state.last_logits in
  let logits = Array2.slice_left logits (Array2.dim1 logits - 1) in
  let candidates = Llama_cpp.Token_data_array.create logits in
  let rec loop state () =
    let last_tokens = Token_buffer.of_seq (Queue.to_seq state.last_tokens) in
    Llama_cpp.sample_repetition_penalty ctx ~candidates ~last_tokens ~penalty:1.1 ;
    let new_token_id = Llama_cpp.sample_token_greedy ctx ~candidates in
    Queue.add new_token_id state.last_tokens ;
    if Int32.equal new_token_id (Llama_cpp.token_eos state.ctx) then
      ( Llama_cpp.batch_free batch ;
        let next_element = (new_token_id, state) in
        Seq.Cons (next_element, Seq.empty) )
    else
      let embd = Token_buffer.of_list [new_token_id] in
      let state = perform_inference state embd batch in
      let next_element = (new_token_id, state) in
      let logits = Option.get state.last_logits in
      let logits = Array2.slice_left logits (Array2.dim1 logits - 1) in
      Llama_cpp.Token_data_array.write_logits candidates logits ;
      Seq.Cons (next_element, loop state)
  in
  loop state
