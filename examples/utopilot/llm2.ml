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
    all_tokens : Token_buffer.t list; (* All chunks of tokens generated so far, most recent chunk on top. *)
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
    all_tokens = [] ;
    last_logits = None ;
    processed_hist = Hash_set.create 41
  }

let ctx = ref None


module State : History.State_S =
struct
  type t = state

  let init () = make_state (Option.get !ctx)
end

module Action : History.Action_S =
struct

  type t = Process_tokens of {
      tokens : Token_buffer.t ;
      n_past : int (* Number of tokens in context just after applying [tokens]. *)
    }

  type nonrec state = state

  let equal (Process_tokens p1) (Process_tokens p2) = 
    Token_buffer.equal p1.tokens p2.tokens && p1.n_past = p2.n_past

  let apply _state _action = assert false

  let undo _state _action =
    (* Check that the action is equal to that on top - otherwise, it is an error. Then remove it.
       The *)
    assert false

  let pp _ _ = ()
end
