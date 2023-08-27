let () =
  print_endline "#include \"llama.h\"";

  Cstubs_structs.write_c Format.std_formatter (module Llama_types.Make)
