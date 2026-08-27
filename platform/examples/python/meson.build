testSuite = ['examples', 'python']

# The extension is staged into the real pypto_serving package, so the repository root is
# what needs to be importable -- exactly what `python -m pypto_serving...` from the repo
# root produces.
repositoryRoot = meson.project_source_root() / '..'

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
		env: {'PYTHONPATH': repositoryRoot},
		depends: [pythonNativeStaged],
		workdir: repositoryRoot,
		timeout: 90,
		suite: testSuite,
	)

	# Spawns its own mpirun per scenario, so it must not be launched under one itself.
	test(
		'pythonFailureModes',
		pythonInstallation,
		args: [
			meson.current_source_dir() + '/failureModes.py',
			mpirunExecutable.full_path(),
			pythonInstallation.full_path(),
			repositoryRoot,
		],
		depends: [pythonNativeStaged],
		workdir: repositoryRoot,
		timeout: 900,
		suite: testSuite,
	)
endif
