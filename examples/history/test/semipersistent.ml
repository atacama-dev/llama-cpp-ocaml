module Action =
struct
  type t =
    | Set of { index : int ; value : int; prev_value : int }

  type state = int array

  let equal a1 a2 = a1 = a2

  let apply array (a : t) =
    match a with
    | Set { index ; value; prev_value = _ } -> array.(index) <- value ; array

  let undo array (a : t) =
    match a with
    | Set { index ; value = _; prev_value } -> array.(index) <- prev_value ; array

  let pp fmtr action =
    match action with
    | Set { index; value; prev_value } ->
      Format.fprintf fmtr "a[%d] = %d -> %d;" index prev_value value
end

let array_length = 10

module State =
struct
  type t = int array
  let init () = Array.init array_length (Fun.const 0)
end

module H = History.Make (State) (Action)

let set handle index value =
  let arr = H.state handle in
  let prev_value = arr.(index) in
  let action = Action.Set { index; value; prev_value } in
  H.process_action action handle

let get handle index =
  let arr = H.state handle in
  arr.(index)

(* -------------------------------------------------------------------------- *)

module Persistent =
struct
  module M = Map.Make (Int)

  let empty = 
    Seq.ints 0 |> Seq.take array_length |> Seq.map (fun i -> (i, 0)) |> M.of_seq

  let get array index = M.find index array

  let set array index value = M.add index value array
end

type action =
  | Set of { index : int ; value : int }
  | Get of { index : int }
  | Save
  | Undo

let index_gen = Crowbar.range array_length
let value_gen = Crowbar.int

let action_gen =
  Crowbar.(map [range 4; index_gen; value_gen]) @@ fun kind index value ->
  match kind with
  | 0 -> Set { index; value }
  | 1 -> Get { index }
  | 2 -> Save
  | 3 -> Undo
  | _ -> assert false

let scenario_gen = Crowbar.list action_gen

let equal oracle handle =
  let array = H.state handle in
  Seq.ints 0 |> Seq.take array_length |> Seq.for_all
  (fun i ->
      array.(i) = Persistent.get oracle i
  )

let rec execute_scenario oracle handle saves acc scenario =
  match scenario with
  | [] -> ()
  | (Set { index; value } as action) :: rest ->
    let oracle = Persistent.set oracle index value in
    let handle = set handle index value in
    if equal oracle handle then
      execute_scenario oracle handle saves (action :: acc) rest
    else
      assert false
  | (Get { index } as action) :: rest ->
    let v_oracle = Persistent.get oracle index in
    let v_handle = get handle index in
    if equal oracle handle && v_oracle = v_handle then
      execute_scenario oracle handle saves (action :: acc) rest
    else
      assert false
  | Save :: rest ->
    execute_scenario oracle handle ((oracle, handle) :: saves) (Save :: acc) rest
  | Undo :: rest ->
    (match saves with
     | [] ->
       execute_scenario oracle handle saves acc rest
     | (oracle, handle) :: saves ->
       execute_scenario oracle handle saves acc rest
    )

let () =
  Crowbar.(add_test ~name:"Semi-persistent array" [scenario_gen]) @@ fun scenario ->
  let oracle = Persistent.empty in
  let handle = H.create () in
  execute_scenario oracle handle [] [] scenario
  

