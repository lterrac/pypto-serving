pythonNativeModule = pythonInstallation.extension_module(
  '_native',
  files('native.cpp'),
  dependencies: [servingBuildDep, servingExamplesHelpersDep, pybind11Dep],
  cpp_args: ['-fvisibility=hidden'],
)

pythonPackageInit = fs.copyfile('__init__.py')
