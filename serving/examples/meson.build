if havePlatformChannels
  subdir('channel_transport')
else
  message('channel_transport skipped: the platform channel headers are incomplete')
endif

if havePlatform
  subdir('coordinator')
else
  message('coordinator example skipped: the platform feature is off')
endif

subdir('routing_modes')

if havePythonBridge and havePythonBindings
  subdir('bindings_embedded')
else
  message('bindings_embedded skipped: it needs both the pythonBridge and pythonBindings features')
endif

if havePythonBridge
  subdir('bridge_embed')
else
  message('bridge_embed skipped: the pythonBridge feature is not enabled')
endif

if haveTokenizers
  subdir('tokenizer_check')
else
  message('tokenizer_check skipped: tokenizers-cpp is not configured')
endif
