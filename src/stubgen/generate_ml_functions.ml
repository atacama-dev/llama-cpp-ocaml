
let main () =
  Cstubs.write_ml Format.std_formatter ~prefix:Sys.argv.(1) (module Llama_functions.Make)

let () = main ()
