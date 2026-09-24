# embedded.cpp is compiled in directly; a static library would drop it.
bindingsEmbeddedExample = executable(
  'bindings-embedded',
  ['main.cpp', bindingsEmbeddedSource],
  dependencies: [bindingsDep, pythonEmbedDep],
  gnu_symbol_visibility: 'hidden',
)
