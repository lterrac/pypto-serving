if havePlatform
  subdir('coordinator')
else
  message('coordinator example skipped: the platform feature is off')
endif

subdir('routing_modes')

if haveTokenizers
  subdir('tokenizer_check')
else
  message('tokenizer_check skipped: tokenizers-cpp is not configured')
endif
