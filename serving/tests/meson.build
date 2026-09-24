gtestDep = dependency('gtest_main', required: true)

servingTestArgs = ['-DSERVING_TEST_FIXTURE_DIR="' + meson.current_source_dir() / 'fixtures' + '"']

servingTestSources = files(
  'config/test_types.cpp',
  'memory/test_kv_cache.cpp',
  'model/test_chat_template.cpp',
)


servingTests = executable(
  'serving-tests',
  servingTestSources,
  cpp_args: servingTestArgs,
  dependencies: [servingBuildDep, gtestDep],
)

test('serving-unit', servingTests, suite: ['unit'], timeout: 120)

# PYTHONPATH: the build root (the extension) and the repository root (the
# parity test).
if havePythonBindings
  pytestProbe = run_command(pythonInstallation, '-c', 'import pytest', check: false)
  if pytestProbe.returncode() == 0
    test(
      'serving-python-bindings',
      pythonInstallation,
      args: ['-m', 'pytest', '-q', '-p', 'no:cacheprovider', meson.current_source_dir() / 'python'],
      env: {'PYTHONPATH': meson.project_build_root() + ':' + meson.project_source_root() / '..'},
      depends: pythonBindings,
      suite: ['python'],
      timeout: 120,
    )
  else
    message('python binding tests skipped: pytest is not importable by ' + pythonInstallation.full_path())
  endif
endif
