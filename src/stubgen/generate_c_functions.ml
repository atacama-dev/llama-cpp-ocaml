let c_headers = "#include \"llama.h\""

let main () =
  print_endline c_headers;
  Cstubs.write_c Format.std_formatter ~prefix:Sys.argv.(1) (module Llama_functions.Make)

let () = main ()
