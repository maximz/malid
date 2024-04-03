"""
When capturing warnings in logging library, allow adding context.

See https://stackoverflow.com/a/68643135/130164 for how to inject context when logging.

This gives us a wrapper around the standard logging library's loggers (produced by logging.getLogger()).

The wrapper provides a context manager that allows adding context to the logger. The context is captured in a log record dictionary called "extra". You can wrap in multiple context managers to add multiple context items.

You must use a log formatter than prints the "extra" dictionary, such as https://github.com/marselester/json-log-formatter

We can use this wrapper instead of standard logging.getLogger loggers. However, to also inject context into warnings, we need to change the "py.warnings" logger into a wrapped logger. We will edit the hook where this logger is retrieved, to instead retrieve a wrapped version of the logger (see https://stackoverflow.com/a/28367973/130164).

We've tested this with parallel execution through joblib.Parallel.
It works for the multiprocessing and threading backends, however we've had mixed results with the loky backend (failed in Jupyter notebook, worked in scripts?).
Loky supposedly does not propagate logging configuration to child processes (see https://github.com/joblib/joblib/issues/1017 and https://stackoverflow.com/a/75935277/130164).
"""

from typing import Optional, Union
import log_with_context
import logging
import warnings
from extendanything import ExtendAnything


class ContextLoggerWrapper(log_with_context.Logger, ExtendAnything):
    """
    Get a logger object:
        from malid.external.logging_context_for_warnings import ContextLoggerWrapper
        logger = ContextLoggerWrapper(name=__name__)
    or if you have an existing logger, you can wrap it (this is identical):
        logger = ContextLoggerWrapper(logger=logging.getLogger(__name__))

    Change root logger to use JSON formatter:
        import json_log_formatter
        logging.getLogger("").handlers[0].setFormatter(
            json_log_formatter.JSONFormatter()
        )

    Optionally also send warnings through this logger:
    (warnings.warn will send log messages to the "py.warnings" logger; this will wrap that logger to allow adding context)
        from malid.external.logging_context_for_warnings import capture_warnings_with_context
        capture_warnings_with_context()

    Log as normal, or with extra:
        logger.info("test", extra={"some_parameter": 5})

    Wrap in context manager (can be nested):
        from log_with_context import add_logging_context
        import warnings
        with add_logging_context(special_parameter="param1"):
            with add_logging_context(special_parameter2="param2"):
                logger.info("test info log")
                warnings.warn("test warning")

    ===

    This is our extension of log_with_context.Logger, adding extra pass-through properties
    We use ExtendAnything for a wildcard passthrough of any unknown attributes to the base logger.
    The required attributes to pass through are:
    - handlers (forwards to self.base_logger.handlers)
    - addHandler (forwards to self.base_logger.addHandler(*args, **kwargs))
    These seem important but not strictly required:
    - name
    - filters
    - propagate
    - parent
    """

    # these will be identical:
    base_logger: logging.Logger  # set by log_with_context.Logger
    _inner: logging.Logger  # set by ExtendAnything

    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[Union[logging.Logger, log_with_context.Logger]] = None,
    ):
        # call parent constructors in a specific order
        log_with_context.Logger.__init__(self, name=name, logger=logger)
        # also store base logger as self._inner
        ExtendAnything.__init__(self, inner=self.base_logger)


def capture_warnings_with_context():
    """Change the "py.warnings" logger into a wrapped logger that allows adding context."""
    # Override warnings.showwarning
    def custom_showwarning(message, category, filename, lineno, file=None, line=None):
        if file is not None:
            if _orig_warnings_showwarning is not None:
                _orig_warnings_showwarning(
                    message, category, filename, lineno, file, line
                )
        else:
            s = warnings.formatwarning(message, category, filename, lineno, line)

            # Get the wrapped logger instead of the standard logger
            logger = ContextLoggerWrapper(logger=logging.getLogger("py.warnings"))
            if not logger.handlers:
                logger.addHandler(logging.NullHandler())
            logger.warning("%s", s)

    # Back up the original showwarning function
    _orig_warnings_showwarning = warnings.showwarning

    # Enable captureWarnings (I think we already have this done, but doesn't hurt to redo)
    logging.captureWarnings(True)

    # Point the warnings.showwarning to our custom function
    warnings.showwarning = custom_showwarning
