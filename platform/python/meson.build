pythonNativeModule = pythonInstallation.extension_module(
  '_native',
  files('native.cpp'),
  dependencies: [servingBuildDep, servingExamplesHelpersDep, pybind11Dep],
  cpp_args: ['-fvisibility=hidden'],
)

# The extension has to land inside the real pypto_serving package: `pypto_serving` has an
# __init__.py, and a regular package always beats an implicit namespace portion whatever the
# sys.path order, so no build-tree mirror is importable once the repository is on the path.
# The staged copy is gitignored, and an editable install resolves to this same directory.
pythonPackageDirectory = meson.project_source_root() / '..' / 'pypto_serving' / 'platform'

pythonNativeStaged = custom_target(
  'stageNativeExtension',
  input: pythonNativeModule,
  output: 'stageNativeExtension.stamp',
  command: [pythonInstallation, files('stageExtension.py'), '@INPUT@', pythonPackageDirectory, '@OUTPUT@'],
  build_by_default: true,
)
