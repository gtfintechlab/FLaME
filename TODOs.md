# TODOs

## Glenn
- `inference.py` should not be in the `together_code` directory ... unless it's not intended to be the entrypoint
- if we maked `superflue.py` the entrypoint for starting the package that makes sense -- in which case the `inference.py` should stay where it is
- the logging level should be determined by the config and then passed down and around through the create logging helper function. each individual file should not be able to set the conditional individually its too messy