type t = {
  root : string ;
  rules : production list
}

and production =
  { name : string;
    rhs : elt list list}

and elt =
  | T_str of string (* string terminal *)
  | T_cset of Uchar.t list (* set of characters *)
  | T_neg_cset of Uchar.t list (* complement of set of characters *)
  | T_range of Uchar.t * Uchar.t (* character range *)
  | T_neg_range of Uchar.t * Uchar.t (* complement of range of characters *)
  (* | Disj of elt * elt (\* disjunction *\) *)
  | NT of { name : string } (* nonterminal *)

module String_set = Set.Make (String)

let used_nonterminals rules =
  List.fold_left (fun acc (production : production) ->
    List.fold_left (fun acc rule ->
        List.fold_left (fun acc elt ->
              match elt with
              | NT { name } -> String_set.add name acc
              | _ -> acc
            ) acc rule
        ) acc production.rhs
    ) String_set.empty rules

let declared_nonterminals root rules =
  List.to_seq rules
  |> Seq.map (fun p -> p.name)
  |> String_set.of_seq
  |> String_set.add root

let make ~root rules =
  let declared = declared_nonterminals root rules in
  let used = used_nonterminals rules in
  if not (String_set.subset used declared) then
    invalid_arg "Grammar.make: ill-formed grammar declaration" ;
  { root; rules }

let production ~name rhs = { name; rhs }

let str s = T_str s
let c u = T_cset [u]
let notc u = T_neg_cset [u]

let range u1 u2 =
  let c = Uchar.compare u1 u2 in
  if c = 1 then
    invalid_arg "Grammar.range: interval is empty" ;
  T_range (u1, u2)

let neg_range u1 u2 =
  let c = Uchar.compare u1 u2 in
  if c = 1 then
    invalid_arg "Grammar.neg_range: interval is empty" ;
  T_neg_range (u1, u2)

let set chars =
  if chars = [] then
    invalid_arg "Grammar.set: empty list" ;
  T_cset chars

let neg_set chars =
  if chars = [] then
    invalid_arg "Grammar.neg_set: empty list" ;
  T_neg_cset chars

let nt name = NT { name }
