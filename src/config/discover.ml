module C = Configurator.V1

let rec link ?(flag = "-l") = function
  | [] -> []
  | lib :: libs -> "-cclib" :: (flag ^ " " ^ lib) :: link ~flag libs

let () =
  C.main ~name:"raylib" (fun c ->
      let library_flags =
        match C.ocaml_config_var c "system" with
        | Some ("linux" | "linux_elf" | "elf") ->
            link [ "llama_bindings" ]
        | Some "macosx" ->
          (* TODO *)
          assert false
        | Some "mingw64" ->
          (* TODO *)
          assert false
        | Some ("netbsd" | "freebsd" | "openbsd" | "bsd" | "bsd_elf") ->
          (* TODO *)
          assert false
        | Some system -> C.die "unsupported system: %s" system
        | None -> C.die "unsupported system"
      in

      C.Flags.write_sexp "library_flags.sexp" library_flags)
