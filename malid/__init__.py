"""Mal-ID."""

from malid import config
from notebooklog import setup_logger

# Get logger for main application (e.g. current notebook) and configure writing to file as well.
# This will capture any direct stdout/stderr writes that didn't come from a logger, too.
logger, log_fname = setup_logger(log_dir=config.paths.log_dir)

# Set up a log sink like Sentry here.
if config.sentry_api_key is not None:
    import sentry_sdk

    sentry_sdk.init(
        config.sentry_api_key,
        # Capture 100% of logs
        traces_sample_rate=1.0,
    )

# Patch some libraries that don't use good logger settings.
# e.g. anndata configures its logger to have a StreamOutput to stderr and not to propagate to the root logger.
import logging

logging.getLogger("anndata").handlers.clear()
logging.getLogger("anndata").addHandler(logging.NullHandler())
logging.getLogger("anndata").propagate = True
# "fontTools.subset" is very verbose on INFO logging level about glyphs
logging.getLogger("fontTools").setLevel(level=logging.WARNING)
