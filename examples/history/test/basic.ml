[@@@ocaml.warning "-32"]

module H = History.Make (struct
    type t = unit
    let init () = ()
  end) (struct 
    type t = string
    type state = unit
    let apply () _ = ()
    let undo () _ = ()
    let equal = String.equal
    let pp = Format.pp_print_string
  end)

let display_graphs = true

let to_dot =
  if display_graphs then
    H.Internal_for_tests.to_dot
  else
    fun _ _ -> ()

let () = Format.printf "@."

let test process =

  let initial = H.create () in

  let chain1 =
    initial
    |> process "A white rabbit"
    |> process " jumped into a hole"
  in

  let () = to_dot [initial] "chain1.dot" in

  let chain2 =
    chain1
    |> process " filled with carrots"
  in

  let () = to_dot [initial] "chain2.dot" in

  let chain3 =
    chain2
    |> process " of all shapes."
  in

  let () = to_dot [initial] "chain3.dot" in

  let chain4 =
    chain1
    |> process " under the tree."
  in

  let () = to_dot [initial; chain1; chain2; chain3; chain4 ] "chain4.dot" in

  let chain5 =
    chain2
    |> process " and other veggies."
  in

  let () = to_dot [initial; chain1; chain2; chain3; chain4; chain5 ] "chain5.dot" in

  let chain6 =
    initial
    |> process "What's up doc"
  in

  let () = to_dot [initial; chain1; chain2; chain3; chain4; chain5; chain6 ] "chain6.dot" in

  let chain7 =
    chain5
    |> process " But he wasn't that hungry."
  in

  let () = to_dot [initial; chain1; chain2; chain3; chain4; chain5; chain6; chain7] "chain7.dot" in

  let chain8 =
    chain6
    |> process ", how about a carrot?"
  in

  let () = to_dot [initial; chain1; chain2; chain3; chain4; chain5; chain6; chain7; chain8] "chain8.dot" in

  ()

let infinite_context _acc _prev_state action =
  Format.printf "Unbounded  \"%s\"@." action ;
  ()

let () = test H.process_action
