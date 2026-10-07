project(
  'PyPTO Serving',
  ['cpp', 'c'],
  default_options: ['cpp_std=c++20', 'buildtype=release'],
)

# `extern/` is on the include path; it is not the meson subproject dir.

####### Warning handling option

fs = import('fs')

warningAsErrorFlags = []
if get_option('compileWarningsAsErrors') == true
  warningAsErrorFlags = ['-Werror']
endif

servingBuildCppArgs = ['-Wfatal-errors'] + warningAsErrorFlags

####### Vendored header-only dependencies
#
# Submodules under `extern/`, as platform/ does for HiCR and TaskR. Each is put
# on the include path at the layout its upstream ships: json exposes
# <nlohmann/json.hpp> from single_include, minja <minja/minja.hpp> from include,
# and cpp-httplib <httplib.h> from its root. minja includes <nlohmann/json.hpp>
# itself, so json must be on the path for minja to compile at all.

servingBuildIncludes = include_directories(
  'include',
  'extern/json/single_include',
  'extern/cpp-httplib',
  'extern/minja/include',
)

servingDependencies = [dependency('threads')]

####### Library sources

servingSources = files(
  'source/infra/state.cpp',
  'source/infra/discovery.cpp',
  'source/infra/capacity.cpp',
  'source/infra/scaling.cpp',
  'source/memory/kv_cache.cpp',
  'source/sched/scheduler.cpp',
  'source/engine/detokenizer.cpp',
  'source/engine/engine.cpp',
  'source/model/chat_template.cpp',
  'source/model/model_files.cpp',
  'source/server/http_server.cpp',
  'source/router/routing.cpp',
  'source/router/strategy.cpp',
  'source/coordinator/module.cpp',
  'source/coordinator/rule_message.cpp',
  'source/coordinator/config_file.cpp',
  'source/coordinator/map_routing.cpp',
  'source/router/proxy.cpp',
  'source/router/router_server.cpp',
)

####### Optional: Python -- the bridge (embedded) and the bindings (extension)
#
# CPython as an embed dependency for the bridge (pythonBridge) and as an
# extension dependency for the bindings (pythonBindings); one installation
# serves both, and either feature alone configures without the other.

havePythonBridge   = false
havePythonBindings = false
pythonEmbedDep     = dependency('threads')
pythonBridgeOption   = get_option('pythonBridge')
pythonBindingsOption = get_option('pythonBindings')
if not pythonBridgeOption.disabled() or not pythonBindingsOption.disabled()
  pythonName = get_option('pythonPath')
  if pythonName == ''
    pythonName = 'python3'
  endif
  pythonModule       = import('python')
  pythonInstallation = pythonModule.find_installation(pythonName, required: pythonBridgeOption.enabled() or pythonBindingsOption.enabled())
  if pythonInstallation.found()
    if not pythonBridgeOption.disabled()
      pythonCandidate = pythonInstallation.dependency(embed: true, required: pythonBridgeOption)
      if pythonCandidate.found()
        pythonEmbedDep   = pythonCandidate
        havePythonBridge = true
      endif
    endif
    if not pythonBindingsOption.disabled()
      if fs.exists('extern/pybind11/include/pybind11/pybind11.h')
        pythonExtDep = pythonInstallation.dependency(embed: false, required: pythonBindingsOption)
        havePythonBindings = pythonExtDep.found()
      elif pythonBindingsOption.enabled()
        error('extern/pybind11 is not checked out: git submodule update --init serving/extern/pybind11')
      else
        message('python bindings skipped: extern/pybind11 is not checked out')
      endif
    endif
  endif
endif

####### Optional: platform/ (the control plane)
#
# serving depends on platform, not the reverse: the coordinator implements
# platform's Module contract.

havePlatform         = false
havePlatformChannels = false
platformBuildDep  = dependency('threads')
####### Optional: CloudR -- the cloud emulator behind the capacity provider
#
# A HiCR backend that emulates instances with arbitrary topologies on one
# machine, so the "ask a provider for a node" path can be exercised without a
# second machine. Needs MPI and hwloc; infra/capacity.hpp stays free of both.
cloudrOption = get_option('cloudr')
haveCloudr   = false
if not cloudrOption.disabled()
  if fs.exists('extern/CloudR/extern/HiCR/meson.build')
    # CloudR does not pull HiCR in when it is itself a subproject: the parent
    # is expected to supply it. platform/ pins the same HiCR commit, so one
    # copy serves both; meson would object loudly if they ever diverged.
    hicrProject = subproject('HiCR', required: cloudrOption,
                             default_options: ['backends=mpi,hwloc,pthreads', 'frontends=RPCEngine,channel'])
    cloudrProject = subproject('CloudR', required: cloudrOption)
    if cloudrProject.found() and hicrProject.found()
      cloudrBuildDep = declare_dependency(dependencies: [cloudrProject.get_variable('CloudRBuildDep'), hicrProject.get_variable('hicrBuildDep')])
      haveCloudr     = true
    endif
  elif cloudrOption.enabled()
    error('extern/CloudR is not checked out: git submodule update --init --recursive serving/extern/CloudR')
  else
    message('cloudr skipped: extern/CloudR is not checked out')
  endif
endif

platformOption    = get_option('platform')
if not platformOption.disabled()
  if fs.exists('subprojects/platform/extern/HiCR/meson.build')
    platformProject  = subproject('platform', required: platformOption, default_options: ['engines=mpi'])
    if platformProject.found()
      platformBuildDep = platformProject.get_variable('servingBuildDep')
      havePlatform     = true
      # platform/system/channels includes <modules/configuration/edge.hpp>, which
      # is not on platform's main; code using a channel is gated on it.
      havePlatformChannels = fs.exists('subprojects/platform/include/modules/configuration/edge.hpp')
      if not havePlatformChannels
        message('platform channels unavailable: modules/configuration/edge.hpp is missing (it is not on main)')
      endif
    endif
  elif platformOption.enabled()
    error('platform/extern/HiCR is not checked out: git submodule update --init platform/extern/HiCR platform/extern/TaskR')
  else
    message('platform skipped: its submodules are not checked out')
  endif
endif

####### Optional: tokenizers-cpp
#
# Found through pkg-config (tokenizers_cpp.pc ships in the project's install
# tree); point PKG_CONFIG_PATH at it, or pass -DtokenizersPath.

tokenizersOption = get_option('tokenizers')
tokenizersDep = dependency('tokenizers_cpp', required: tokenizersOption)
haveTokenizers = tokenizersDep.found()

####### Creating the serving dependency

if haveTokenizers
  servingSources += files('source/model/hf_tokenizer.cpp')
  servingDependencies += tokenizersDep
endif

if havePythonBridge
  servingSources += files('source/engine/bridge_executor.cpp')
  servingDependencies += pythonEmbedDep
endif

if haveCloudr
  servingSources      += files('source/infra/cloudr_capacity.cpp')
  servingDependencies += cloudrBuildDep
  servingBuildCppArgs += ['-DSERVING_HAVE_CLOUDR=1']
endif

servingLib = static_library(
  'serving',
  servingSources,
  cpp_args: servingBuildCppArgs,
  include_directories: servingBuildIncludes,
  dependencies: servingDependencies,
)

servingBuildDep = declare_dependency(
  compile_args: servingBuildCppArgs,
  include_directories: servingBuildIncludes,
  dependencies: servingDependencies,
  link_with: servingLib,
)

####### The Python bindings
#
# The binding body is built into the extension module and, through
# embedded.cpp, into the serving binary. pybind11 needs exceptions and hidden
# visibility.

if havePythonBindings
  pybind11Includes       = include_directories('extern/pybind11/include')
  bindingsCppArgs        = ['-fexceptions']
  bindingsEmbeddedSource = files('source/bindings/embedded.cpp')

  # HfTokenizer is bound only where tokenizers-cpp is configured.
  bindingsCppArgs += ['-DSERVING_HAVE_TOKENIZERS=' + (haveTokenizers ? '1' : '0')]

  bindingsCore = static_library(
    'serving-bindings',
    files(
      'source/bindings/bindings.cpp',
      'source/bindings/infra_bindings.cpp',
      'source/bindings/config_bindings.cpp',
      'source/bindings/template_bindings.cpp',
      'source/bindings/kv_cache_bindings.cpp',
      'source/bindings/scheduler_bindings.cpp',
      'source/bindings/tokenizer_bindings.cpp',
      'source/bindings/engine_bindings.cpp',
      'source/bindings/server_bindings.cpp',
    ),
    cpp_args: bindingsCppArgs,
    include_directories: pybind11Includes,
    dependencies: [servingBuildDep, pythonExtDep],
    gnu_symbol_visibility: 'hidden',
    pic: true,
  )

  bindingsDep = declare_dependency(
    compile_args: bindingsCppArgs,
    include_directories: pybind11Includes,
    dependencies: [servingBuildDep],
    link_with: bindingsCore,
  )

  pythonBindings = pythonInstallation.extension_module(
    '_pypto_serving',
    files('source/bindings/module.cpp'),
    dependencies: [bindingsDep, pythonExtDep],
    gnu_symbol_visibility: 'hidden',
    install: true,
  )
endif

####### The serving binary
#
# Needs the Python bridge and tokenizers-cpp.

if havePythonBridge and haveTokenizers
  servingBinarySources      = ['source/main.cpp']
  servingBinaryDependencies = [servingBuildDep]
  # The embedded interpreter can import _pypto_serving.
  if havePythonBindings
    servingBinarySources      += bindingsEmbeddedSource
    servingBinaryDependencies += bindingsDep
  endif
  servingBinary = executable(
    'pypto-serving-cpp',
    servingBinarySources,
    dependencies: servingBinaryDependencies,
    gnu_symbol_visibility: havePythonBindings ? 'hidden' : 'default',
    install: true,
  )
else
  message('pypto-serving-cpp skipped: it needs both the pythonBridge and tokenizers features')
endif

####### The router binary
#
# No Python and no tokenizer: the router never parses a payload.

routerBinary = executable(
  'pypto-serving-router-cpp',
  ['source/router_main.cpp'],
  dependencies: [servingBuildDep],
  install: true,
)

####### Sub-directories

if meson.is_subproject() == false
  if get_option('buildTests')
    subdir('tests')
  endif
  if get_option('buildExamples')
    subdir('examples')
  endif
endif

summary(
  {
    'platform': havePlatform,
    'platform channels': havePlatformChannels,
    'python bridge': havePythonBridge,
    'python bindings': havePythonBindings,
    'tokenizers-cpp': haveTokenizers,
    'tests': get_option('buildTests'),
    'examples': get_option('buildExamples'),
  },
  section: 'PyPTO Serving',
)
