project(
  'PyPTO Serving Platorm',
  ['cpp', 'c'],
  subproject_dir: 'extern',
  default_options: ['cpp_std=c++20', 'buildtype=release'],
)

####### Storage for serving dependencies

# Warning handling option
warningAsErrorFlags = []
if get_option('compileWarningsAsErrors') == true
  warningAsErrorFlags = ['-Werror']
endif

servingDependencies = []
servingBuildCppArgs = ['-Wfatal-errors'] + warningAsErrorFlags
servingBuildIncludes = include_directories(['include'])

# Getting selected distributed engines

engines = get_option('engines')

# Storing required HiCR backends and frontends
additionalHiCRBackends = []
additionalHiCRFrontends = []

####### Getting HiCR dependency

if 'mpi' in engines
  if 'mpi' not in additionalHiCRBackends
    additionalHiCRBackends += ['mpi']
  endif
endif

# Selecting default HiCR Backends
HiCRBackends = ['hwloc', 'pthreads', 'boost']

# adding additional backends but only if not already listed
foreach b : additionalHiCRBackends
  if b not in HiCRBackends
    HiCRBackends += [b]
  endif
endforeach

HiCRFrontends = [ ] + additionalHiCRFrontends
HiCRProject = subproject(
  'HiCR',
  required: true,
  default_options: ['backends=' + ','.join(HiCRBackends), 'frontends=' + ','.join(HiCRFrontends)],
)
HiCRBuildDep = HiCRProject.get_variable('hicrBuildDep')
servingDependencies += HiCRBuildDep

if 'mpi' in engines
  mpirunExecutable = HiCRProject.get_variable('mpirunExecutable')
endif

####### Getting TaskR dependency

TaskRProject = subproject('TaskR', required: true, default_options: [])
TaskRBuildDep = TaskRProject.get_variable('TaskRBuildDep')
servingDependencies += TaskRBuildDep

####### Creating serving dependency

servingBuildDep = declare_dependency(
  compile_args: servingBuildCppArgs,
  include_directories: servingBuildIncludes,
  dependencies: servingDependencies,
)

# Helpers shared by the examples and by the Python bindings.
# TODO: these belong in include/ rather than examples/include/ now that non-example code uses them.
servingExamplesHelpersDep = declare_dependency(include_directories: include_directories('examples/include'))

####### Python bindings (optional)

if get_option('buildPythonBindings')
  fs = import('fs')
  pythonModule = import('python')
  pythonInstallation = pythonModule.find_installation(get_option('pythonInterpreter'), required: true)

  # pybind11 comes in through TaskR -> TraCR; it is header-only, so no extra build is needed.
  # Chosen over nanobind purely because it is already vendored in this tree.
  pybind11Dep = declare_dependency(
    include_directories: include_directories('extern/TaskR/extern/tracr/extern/pybind11/include'),
  )

  subdir('python')
endif

if meson.is_subproject() == false
  if get_option('buildTests')
    subdir('examples')
  endif
endif

