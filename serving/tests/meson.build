gtestDep = dependency('gtest_main', required: true)

servingTestArgs = ['-DSERVING_TEST_FIXTURE_DIR="' + meson.current_source_dir() / 'fixtures' + '"']

servingTestSources = files(
  'infra/test_state.cpp',
  'infra/test_document.cpp',
  'infra/test_capacity.cpp',
  'infra/test_scaling.cpp',
  'infra/test_launcher.cpp',
  'infra/test_monitor.cpp',
  'infra/test_deployment_server.cpp',
  'memory/test_kv_cache.cpp',
  'sched/test_scheduler.cpp',
  'engine/test_detokenizer.cpp',
  'engine/test_engine.cpp',
  'server/test_http_server.cpp',
  'router/test_routing.cpp',
  'router/test_strategy.cpp',
  'coordinator/test_coordinator.cpp',
  'coordinator/test_rule_message.cpp',
  'coordinator/test_config_file.cpp',
  'coordinator/test_map_routing.cpp',
  'replica/test_rule_table.cpp',
  'replica/test_pipeline.cpp',
  'router/test_router_server.cpp',
  'deployer/test_deployment.cpp',
  'model/test_chat_template.cpp',
)

if haveTokenizers
  servingTestSources += files('model/test_hf_tokenizer.cpp')
endif

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
