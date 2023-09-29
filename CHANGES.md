# 0.0.2
- bump `llama.cpp` version, breaking API changes in inference
- new module [BNF] implementing a basic high-level API to define grammars
- new module [Token_buffer] with conversion functions to Seq, array and iterators.
- bugfix: `sample_token` was mapped to `sample_token_greedy`
- bugfix: a spurious null character was added at the end of tokenized text

# 0.0.1
- First release of `llama-cpp-ocaml`, ctypes bindings for https://github.com/ggerganov/llama.cpp/
