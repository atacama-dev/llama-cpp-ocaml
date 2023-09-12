let c_headers = "#include <string.h>\n#include \"llama.h\"\n#include \"extra_stubs.h\"\n"

let main () =
  print_endline c_headers;
  Cstubs.write_c Format.std_formatter ~prefix:Sys.argv.(1) (module Llama_functions.Make)

let () = main ()
