(*
 * prompt_repl.ml
 * --------
 * Copyright : (c) 2023, Ilias Garnier <igarnier@protonmail.com>
 *
 * Copyright : (c) 2015, Martin DeMello <mdemello@google.com>
 * Licence   : BSD3
 * Copied from the lambda-term distribution
 *)

(* Add a REPL to an existing interpreter *)

open React
open Lwt
open LTerm_text
open Bigarray

module Token_buffer = Llama_cpp.Token_buffer

(* +-----------------------------------------------------------------+
   | Interpreter                                                     |
   +-----------------------------------------------------------------+ *)

module Interpreter = struct

  type token = Llama_cpp.token

  type state =
    { n : int ; (* Index of repl interaction *)
      ctx : Llama_cpp.context;
      n_keep : int ; (* Number of tokens to keep when resetting context *)
      n_batch : int ; (* Max size of a batch *)
      n_threads : int ; (* Number of parallel threads to use to perform inference. *)
      n_max_samples : int ; (* Maximum sampling budget per user input *)
      n_past : int ;
      embd : token list;
      last_tokens : token list ;
      last_logits : Llama_cpp.logits option ;
      grammar : Llama_cpp.grammar option
    }

  let make_state ?(n_keep = 32) ?(n_batch = 32) ?(n_threads = 8) ?(n_max_samples = 512) ?grammar ctx =
    {
      n = 1 ;
      ctx ;
      n_keep ;
      n_batch ;
      n_threads ;
      n_max_samples ;
      n_past = 0 ;
      embd = [] ;
      last_tokens = [];
      last_logits = None ;
      grammar
    }

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

  let grammar =
    let open Llama_cpp.BNF in
    make ~root:"root"
      [
        production ~name:"root" [
          [c (Uchar.of_char 't'); c (Uchar.of_char 'e'); c (Uchar.of_char 's'); nt "root"] ;
        ]
      ]


  (*
     Context swapping is used when the context is too small to fit the next sequence of tokens.
     Invariant: [embd < n_ctx] where [n_ctx] is the total size of the context

     Note: in llama.cpp/examples/main, `last_tokens` always contains `embd` as a prefix.
     Here, `last_tokens` only contain tokens that have been sampled.

     In the following diagram, tokens are in 'chronological' order (reading left to right):
     - processed tokens are in the order they were processed
     - tokens to be processed are in the order they are going to be processed

                             <--- processed tokens      now     tokens to be processed ->
                                                         |
                                         n_past          |
     context:                    kkkkkkkk----------------|_______________
     embd:                                               |xxxxxxxxxxxxxxxxxxxx
     last_tokens: ---------------kkkkkkkkdddddddmmmmmmmmm|

     where
     - k = kept tokens (won't need to be re-sampled)
     - m = moved tokens (will need to be re-sampled)
     - d = dropped tokens (will disappear from the context)

     After context swapping:
                                         n_past          |
     context:                                    kkkkkkkk|_______________________________
     embd:                                               |mmmmmmmmmxxxxxxxxxxxxxxxxxxxx
     last_tokens: ---------------kkkkkkkkdddddddmmmmmmmmm|

   *)
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

  let tokens_to_string state tokens =
    let buff = Buffer.create 1024 in
    List.iter (fun token ->
        (match Llama_cpp.token_to_piece state.ctx token with
         | Ok s -> Buffer.add_string buff s
         | Error `Invalid_token -> failwith "Invalid token")
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

  (* This produces a transient sequence because of the shared [candidate] array *)
  let step ~add_bos state prompt =
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
      Option.iter (fun grammar -> Llama_cpp.sample_grammar ctx ~candidates grammar) state.grammar ;
      let new_token_id = Llama_cpp.sample_token_greedy ctx ~candidates in
      Option.iter (fun grammar ->
          Llama_cpp.grammar_accept_token ctx grammar new_token_id
        ) state.grammar ;
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
end

(* +-----------------------------------------------------------------+
   | Prompt and output wrapping                                      |
   +-----------------------------------------------------------------+ *)

(* Create a prompt based on the current interpreter state *)
let make_prompt state =
  let prompt = Printf.sprintf "Prompt [%d]: " state.Interpreter.n in
  eval [ S prompt ]

(* +-----------------------------------------------------------------+
   | Customization of the read-line engine                           |
   +-----------------------------------------------------------------+ *)

class read_line ~term ~history ~state = object(self)
  inherit LTerm_read_line.read_line ~history ()
  inherit [Zed_string.t] LTerm_read_line.term term

  method! show_box = false

  initializer
    self#set_prompt (S.const (make_prompt state))
end

(* +-----------------------------------------------------------------+
   | Main loop                                                       |
   +-----------------------------------------------------------------+ *)

let (let*) = (>>=)

let (let+) = (>|=)

let token_to_string (state : Interpreter.state) token =
  match Llama_cpp.token_to_piece state.ctx token with
  | Ok s -> s
  | Error `Invalid_token -> failwith "Invalid token"

(* Sample from the LLM until [not continue] or we exhaust the sampling budget.
   We return the final state upon termination. *)
let rec sampling_loop continue budget term prev_state seq : Interpreter.state Lwt.t =
  if not !continue || budget <= 0 then
    let* () = LTerm.fprint term "<generation interrupted>\n" in
    Lwt.return prev_state
  else
    match Seq.uncons seq with
    | None -> Lwt.return prev_state
    | Some ((token, state), rest) ->
      let output = token_to_string state token in
      let* () = Lwt.pause () in
      let* () = LTerm.fprint term output in
      let* () = LTerm.flush term in
      sampling_loop continue (budget - 1) term state rest

let rec call_when_ctrl_c term k =
  let* ev = LTerm.read_event term in
  match ev with
  | LTerm_event.Resize _
  | LTerm_event.Sequence _
  | LTerm_event.Mouse _ ->
    call_when_ctrl_c term k
  | LTerm_event.Key { control; meta = _; shift = _; code } ->
    if control && code = Char (Uchar.of_char 'c') then
      k `Stop
    else
      call_when_ctrl_c term k

let sampling term prev_state seq =
  let* mode = LTerm.enter_raw_mode term in
  let continue  = ref true in
  let poll = call_when_ctrl_c term (function `Stop -> continue := false ; Lwt.return_unit) in
  let state = sampling_loop continue prev_state.Interpreter.n_max_samples term prev_state seq in
  let* () = Lwt.join [poll; Lwt.map ignore state] in
  let* () = LTerm.leave_raw_mode term mode in
  state

let rec loop ~add_bos term history state =
  Lwt.catch (fun () ->
      let rl = new read_line ~term ~history:(LTerm_history.contents history) ~state in
      rl#run >|= fun command -> Some command)
    (function
      | Sys.Break -> return None
      | exn -> Lwt.fail exn)
  >>= function
  | Some command ->
    let command_utf8 = Zed_string.to_utf8 command in
    let state_seq = Interpreter.step ~add_bos state command_utf8 in
    let* state = sampling term state state_seq in
    LTerm_history.add history command;
    let state = { state with n = state.n + 1 } in
    loop ~add_bos:false term history state
  | None ->
    loop ~add_bos term history state

(* +-----------------------------------------------------------------+
   | Entry point                                                     |
   +-----------------------------------------------------------------+ *)

let print_info term =
  let* () = LTerm.fprint term "Use ctrl-c to interrupt prompt generation. LLM context is kept across interruptions.\n" in
  LTerm.fprint term "Use ctrl-d to quit.\n"

let main model =
  let* () = LTerm_inputrc.load () in
  Lwt.catch (fun () ->
      Llama_cpp.log_set (fun _log_level _msg -> ()) ;
      let grammar = Llama_cpp.grammar_from_bnf Interpreter.grammar in
      let ctx_params = Llama_cpp.Context_params.default () in
      let model =
        match Llama_cpp.load_model_from_file model ctx_params with
        | None ->
          Printf.eprintf "%s: error: unable to load model\n" __FUNCTION__ ;
          exit 1
        | Some model -> model
      in
      let ctx = Llama_cpp.new_context_with_model model ctx_params in
      let state = Interpreter.make_state ~grammar ctx in
      let* term = Lazy.force LTerm.stdout in
      let* () = print_info term in
      loop ~add_bos:true term (LTerm_history.create []) state)
    (function
      | LTerm_read_line.Interrupt -> Lwt.return ()
      | exn -> Lwt.fail exn)

let usage () =
  Format.eprintf "usage: prompt_repl.exe model_file@." ;
  exit 1

let () =
  match Array.to_list Sys.argv |> List.tl with
  | [model_file]->
    Lwt_main.run (main model_file)
  | _ ->
    usage ()
