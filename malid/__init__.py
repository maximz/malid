"""Mal-ID."""

from malid import config
from notebooklog import setup_logger
import logging
import json_log_formatter
from malid.external.logging_context_for_warnings import (
    ContextLoggerWrapper,
    capture_warnings_with_context,
)

# Get logger for main application (e.g. current notebook) and configure writing to file as well.
# This will capture any direct stdout/stderr writes that didn't come from a logger, too.
logger, log_fname = setup_logger(log_dir=config.paths.log_dir)

# Wrap the logger in a context logger that allows adding context to log messages.
# We'll be able to do:
# from log_with_context import add_logging_context
# with add_logging_context(special_parameter="param1"):
#   malid.logger.info("test")
logger = ContextLoggerWrapper(logger=logging.getLogger(__name__))

# Change root logger to use JSON formatter (required to see extra context parameters in log messages)
logging.getLogger("").handlers[0].setFormatter(json_log_formatter.JSONFormatter())

# Send warnings through a wrapper logger
# (warnings.warn will send log messages to the "py.warnings" logger; this will wrap that logger to allow adding context)
capture_warnings_with_context()

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

logging.getLogger("anndata").handlers.clear()
logging.getLogger("anndata").addHandler(logging.NullHandler())
logging.getLogger("anndata").propagate = True

# Turn down verbosity on some loggers
for logger_name in [
    # "fontTools.subset" is very verbose on INFO logging level about glyphs
    "fontTools",
    # Dask loggers are also extremely verbose on INFO logging level
    "distributed.nanny",
    "distributed.scheduler",
    "distributed.core",
    "distributed.utils_perf",
]:
    logging.getLogger(logger_name).setLevel(level=logging.WARNING)
