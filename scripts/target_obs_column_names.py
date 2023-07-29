from malid import config

for target_obs_column in config.classification_targets:
    print(target_obs_column.name)
