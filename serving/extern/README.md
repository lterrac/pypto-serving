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
