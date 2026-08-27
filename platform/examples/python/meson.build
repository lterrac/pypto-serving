testSuite = ['examples', 'python']

# The extension is built into <builddir>/python/pypto_serving/platform/, mirroring the
# package it must be importable as. Putting that directory on PYTHONPATH is all it takes.
pythonBindingsPath = meson.project_build_root() / 'python'

if get_option('buildTests')
	test(
		'pythonRoundTrip',
		mpirunExecutable,
		args: [
			'-np', '2',
			'--oversubscribe',
			'-x', 'PYTHONPATH',
			pythonInstallation.full_path(),
			meson.current_source_dir() + '/roundTrip.py',
			meson.current_source_dir() + '/policy.json',
		],
		env: {'PYTHONPATH': pythonBindingsPath},
		depends: [pythonNativeModule, pythonPackageInit],
		timeout: 90,
		suite: testSuite,
	)
endif
