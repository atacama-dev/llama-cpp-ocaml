(** The module type of actions. *)
module type Action_S =
sig

  (** The type of actions. *)
  type t

  (** The type of states on which actions are performed. *)
  type state

  (** [equal] tests whether two actions are equal. *)
  val equal : t -> t -> bool

  (** [apply state a] applies [a] on [state]. *)
  val apply : state -> t -> state

  (** [undo state a] reverts the effect of [a] on [state]. 
      We expect that [undo (apply state a) a] is equal to [state]: however,
      it is not a requirement of this library. *)
  val undo : state -> t -> state

  (** [pp fmtr a] prints the action [a]. *)
  val pp : Format.formatter -> t -> unit
end

(** The module type of states. *)
module type State_S =
sig

  (** The type of states. *)
  type t

  (** [init ()] is an empty state. *)
  val init : unit -> t
end

module type S =
sig
  type action
  type state

  (** The type of handles to states. *)
  type t

  (** Create a handle. The associate state is obtained by [State_S.init]. *)
  val create : unit -> t

  (** [reroot handle] let [handle] become the current holder of the state. 
      Calling [reroot] may trigger undoing and redoing of some actions. *)
  val reroot : t -> unit

  (** [process_action a handle] applies [a] on the state corresponding to [handle]. *)
  val process_action : action -> t -> t

  (** [state handle] returns the state corresponding to [handle]. *)
  val state : t -> state

  module Internal_for_tests :
  sig
    val to_dot : t list -> string -> unit
  end
end

module Make
    (State : State_S)
    (Action : Action_S with type state = State.t)
  : S with type action = Action.t and type state = State.t
=
struct
  type action = Action.t
  type state = State.t

  type 'a trace = { mutable elt : 'a; mutable next : 'a node }

  and 'a node =
    | Top of state
    | Node of 'a trace

  type desc =
    | Empty
    | Done of { uid : int ;
                action : action }
    | Redo of { uid : int ;
                action : action }

  type t = desc trace ref

  let gen =
    let x = ref 0 in
    fun () ->
      let v = !x in
      incr x;
      v

  let rec get_acc_and_reverse prev_node node_opt k =
    match node_opt with
    | Top state ->
      k state
    | Node node ->
      match node.elt with
      | Empty | Redo _ ->
        (* Impossible because a [Done] node may only point to [Top] or [Done] node. *)
        assert false
      | Done { uid = _ ; action } ->
        get_acc_and_reverse node node.next (fun state ->
            let state = Action.undo state action in
            let result = k state in
            node.elt <- Redo { uid = gen () ; action } ;
            node.next <- (Node prev_node) ;
            result
          )

  let create () =
    ref { elt = Empty ;
          next = Top (State.init ()) }

  let rec reroot (base : t) : unit =
    let trace = !base in
    match trace.elt with
    | Empty | Done _ ->
      (* Invariant: at this point, the chain of nodes reachable from [next] only contains
         [Done] nodes. *)
      let state = get_acc_and_reverse trace trace.next Fun.id in
      trace.next <- Top state
    | Redo _ ->
        (*
         ... -> root <- redo( action0) <- redo( action1) <- redo( action2) <- ... <- base=redo( actionN)
                 |
                 \-> Done( action0) -> Done( action1) -> Done( action2' <>  action2) -> ... -> Top
                                     ^^^^^^^^^^^
                                     node on which we recursively call [process_action]
         *)
      let rec get_root node acc =
        match node.elt with
        | Empty
        | Done _ ->
          max_prefix node node.next acc
        | Redo { uid = _; action } ->
          (match node.next with
           | Top _ ->
             (* Invariant: a [Redo] node {b must} point to a [Done] or an [Empty] node. *)
             assert false
           | Node next ->
             get_root next (action :: acc)
          )
      and max_prefix prev_node node_opt acc =
        match node_opt, acc with
        | _, [] ->
          (* Impossible: [get_root] is called from a [Redo] node. *)
          assert false
        | Top _state, _ ->
          (* TODO (to test!):
             - we have to update [base] both in this case and in the [Done] case below, no?
          *)
          replay prev_node acc
        | Node node, action' :: acc' ->
          (match node.elt with
           | Empty | Redo _ ->
             (* Impossible because [Done] nodes cannot point to [Redo] nodes. *)
             assert false
           | Done { uid = _; action } ->
             if Action.equal action action' then
               max_prefix node node.next acc'
             else
               replay prev_node acc
          )
      and replay node acc =
        let replayed =
          List.fold_left (fun node action_to_redo ->
              process_action action_to_redo node
            ) (ref node) acc
        in
        base := !replayed
      in
      get_root trace []

  and process_action new_action (base : t) : t =
    reroot base ;
    let node = !base in
    match node.next with
    | Top state ->
      let state' = Action.apply state new_action in
      let trace = { elt =
                      Done { uid = gen () ;
                             action = new_action } ;
                    next = Top state' } in
      node.next <- Node trace ;
      ref trace
    | _ ->
      assert false
 
  let state base =
    reroot base ;
    let node = !base in
    match node.next with
    | Top state -> state
    | _ ->
      assert false
 
  module Internal_for_tests =
  struct
    module G = Graph.Pack.Digraph

    let vertex_labels = Hashtbl.create 11

    let end_ = G.V.create @@ gen ()

    let start = G.V.create @@ gen ()

    let () =
      Hashtbl.add vertex_labels start `Start ;
      Hashtbl.add vertex_labels end_ `End

    module Display = struct
      include G

      let vertex_name v = G.V.label v |> string_of_int

      let graph_attributes _ = []
      let default_vertex_attributes _ = []
      let vertex_attributes v =
        match Hashtbl.find_opt vertex_labels v with
        | None -> []
        | Some `Start ->
          [`Label "START"]
        | Some `End ->
          [`Label "EOS"]
        | Some (`Done action) ->
          let s = Format.asprintf "%a" Action.pp action in
          [`Label s; `Shape `Box]
        | Some (`Redo action) ->
          let s = Format.asprintf "%a" Action.pp action in
          [`Label s; `Shape `Diamond]

      let default_edge_attributes _ = []
      let edge_attributes e = [ `Label (string_of_int (E.label e) ) ]
      let get_subgraph _ = None
    end
    module Dot_ = Graph.Graphviz.Dot(Display)
    module Neato = Graph.Graphviz.Neato(Display)

    let dot_output g f =
      let oc = open_out f in
      if G.is_directed then Dot_.output_graph oc g else Neato.output_graph oc g;
      close_out oc

    let display_with_png g file =
      let tmp = Filename.temp_file "graph" ".dot" in
      dot_output g tmp;
      ignore (Sys.command ("dot -Tpng " ^ tmp ^ " > " ^ file));
      Sys.remove tmp

    let add_vertex g v =
      if G.mem_vertex g v then () else G.add_vertex g v

    let add_edge g v1 v2 =
      if G.mem_edge g v1 v2 then () else G.add_edge g v1 v2

    let to_dot (handles : t list) filename =
      let g = G.create () in
      add_vertex g end_ ;

      let visited = Hashtbl.create 11 in

      let rec unfold node_opt =
        match node_opt with
        | Top _ -> end_
        | Node node ->
          (match node.elt with
           | Empty ->
             let next = unfold node.next in
             add_edge g start next ;
             start
           | Done { uid; action; _ } -> (
               match Hashtbl.find_opt visited uid with
               | Some label -> label
               | None ->
                 (let label = G.V.create uid in
                  Hashtbl.add visited uid label ;
                  add_vertex g label ;
                  let next = unfold node.next in
                  add_edge g label next ;
                  Hashtbl.add vertex_labels label (`Done action) ;
                  label))
           | Redo { uid ; action; _ } -> (
               match Hashtbl.find_opt visited uid with
               | Some label -> label
               | None ->
                 (let label = G.V.create uid in
                  Hashtbl.add visited uid label ;
                  add_vertex g label ;
                  let next = unfold node.next in
                  add_edge g label next ;
                  Hashtbl.add vertex_labels label (`Redo action) ;
                  label)))
      in
      List.iter (fun node_ref -> ignore @@ unfold (Node !node_ref)) handles ;
      dot_output g filename ;
      display_with_png g (filename ^ ".png")
  end

end
