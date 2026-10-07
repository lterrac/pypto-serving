# Third-party dependencies

Git submodules, as `platform/extern/` does for HiCR and TaskR. Header-only; on the
include path via `servingBuildIncludes` in `serving/meson.build`.

| Path | Upstream | Pin | Licence |
| --- | --- | --- | --- |
| `minja/` | https://github.com/google/minja | `021c2293c187` | MIT |
| `json/` | https://github.com/nlohmann/json | `v3.12.0` | MIT |
| `cpp-httplib/` | https://github.com/yhirose/cpp-httplib | `v0.43.1` | MIT |
| `pybind11/` | https://github.com/pybind/pybind11 | `v3.1.0` | BSD-3 |

```
git submodule update --init serving/extern
```

`json`, `cpp-httplib` and `pybind11` are shallow (`.gitmodules`): only the pinned tip
is needed.

minja is upstream, not llama.cpp's copy: that copy's `in` operator has no string
containment, so Qwen3's `{%- if '</think>' in content %}` never fires and reasoning
blocks leak into the prompt. `tests/model/test_chat_template.cpp` checks rendering
against the Python output.

## CloudR

`CloudR` is a HiCR-based cloud emulator: it creates instances with arbitrary
topologies on one machine. It backs `infra/cloudr_capacity.hpp`, which is how
the "ask a provider for a node" half of M2 is exercised without a second
machine. Behind `-Dcloudr=`; the default build links no MPI.

It does not pull HiCR in when used as a subproject, so `serving/meson.build`
supplies it from `extern/CloudR/extern/HiCR`. That is the same commit
`platform/` pins, so one copy serves both.
