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

# build_always_stale because the real output lands in the source tree, outside anything
# ninja tracks: the declared output is only a stamp, so without this, deleting the staged
# .so leaves ninja reporting "no work to do" -- and with two build trees configured, the
# second one would silently leave the first one's binary in place to be tested.
pythonNativeStaged = custom_target(
  'stageNativeExtension',
  input: pythonNativeModule,
  output: 'stageNativeExtension.stamp',
  command: [pythonInstallation, files('stageExtension.py'), '@INPUT@', pythonPackageDirectory, '@OUTPUT@'],
  build_by_default: true,
  build_always_stale: true,
)
